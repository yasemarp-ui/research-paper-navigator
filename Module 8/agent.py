"""Research Paper Navigator agent: semantic cache -> router -> (direct answer | LangGraph agent).

Adapted from the Lab 7.2 solution (Module 7/lab_7_2_efficient_agent_solution.py, left
unchanged). The cache, router, satisfaction check, gpt-5.4 fallback and per-model token
tracking are the lab's. What changed for the paper collection:
  * Retrieval: HybridRetriever (chunks, fused by chunk_id) and PaperIndex (catalog lookup)
    from retrieval.py replace the e-mail DB, and a find_paper tool replaces the by_date tool.
  * Prompts: paper-specific (prompts.py), answers cite file name + page.
  * Safety (Capstone Checkpoint 7.1): every user turn — the question and any clarification
    answer — is sanitized before it reaches the cache, the retriever or a prompt, and every
    prompt keeps its parts inside escaped XML tags.
  * Cost: the planner sees short chunk previews instead of whole documents, and the answer
    step sees at most config.ANSWER_MAX_CHUNKS chunks.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Optional, TypedDict

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import config
from provenance import Provenance
from prompts import ANSWER_SYSTEM, CACHE_VALIDATE_SYSTEM, PLAN_SYSTEM, ROUTER_SYSTEM, SATISFACTION_SYSTEM
from retrieval import HybridRetriever, PaperIndex, get_embeddings
from safety import escape_attr, escape_xml, sanitize_user_text, was_sanitized, xml_prompt


def make_chat_llm(model: str, reasoning_effort: str | None = None) -> ChatOpenAI:
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return ChatOpenAI(model=model, api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url=config.OPENROUTER_BASE_URL, timeout=120, max_retries=2, **extra)


def estimate_cost_usd(usage: dict[str, dict[str, int]]) -> float:
    """Rough USD estimate from a token tally (from Lab 7.2). Cached tokens are counted at the
    full input rate, so this is an upper-ish estimate; unknown models contribute 0."""
    total = 0.0
    for model, counts in usage.items():
        price = config.APPROX_PRICE_PER_MTOK.get(model)
        if price:
            total += counts["input"] / 1e6 * price["input"] + counts["output"] / 1e6 * price["output"]
    return total


# Citation tags the model writes: [S3], [P1], [S1, S4] (passages S, catalog records P).
_TAG_GROUP_RE = re.compile(r"\[\s*([SP]\d+(?:\s*[,;/–-]\s*[SP]\d+)*)\s*\]")
_TAG_RANGE_RE = re.compile(r"([SP])(\d+)\s*[–-]\s*[SP](\d+)")


def _split_tags(group: str) -> list[str]:
    """Ids of one bracket; the model sometimes writes a range such as P1–P20."""
    tags: list[str] = []
    for part in re.split(r"\s*[,;/]\s*", group):
        span = _TAG_RANGE_RE.fullmatch(part)
        tags += ([f"{span[1]}{n}" for n in range(int(span[2]), int(span[3]) + 1)] if span else [part])
    return tags

# An answer that says the evidence lacks the answer is never accepted on the direct path:
# nano judged such answers satisfactory in testing, which skipped the stronger fallback.
_NOT_FOUND_RE = re.compile(
    r"(?i)\b(can(?:no|'|’)t|could ?n[o'’]t|unable to)\b[^.\n]{0,40}\b(find|determine|say|answer|locate|identify)\b"
    r"|\b(do|does|did) ?n[o'’]t (contain|include|mention|define|describe|provide|say)\b"
    r"|\bno (information|mention|definition|evidence)\b|\bnot (found|mentioned|covered|available)\b"
)


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ─── Semantic cache (from Lab 7.2) ───────────────────────────────────
class _SemanticCache:
    """Cosine-similarity cache of (question -> answer) pairs; evicts the oldest entry when full.
    Same embedding model as retrieval, but a much higher threshold: returning a wrong cached
    answer is worse than paying to recompute."""

    def __init__(self, cache_dir=config.CACHE_DIR, max_size: int = config.CACHE_MAX_SIZE, debug: bool = False):
        self._client = chromadb.PersistentClient(path=str(cache_dir),
                                                 settings=chromadb.Settings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection("response_cache", metadata={"hnsw:space": "cosine"})
        self._embedder = get_embeddings()
        self._max_size = max_size
        self._debug = debug

    def lookup(self, question: str, threshold: float) -> Optional[tuple[str, float]]:
        """Return (cached_answer, similarity) if a close-enough match exists, else None."""
        if self._col.count() == 0:
            return None
        results = self._col.query(query_embeddings=[self._embedder.embed_query(question)], n_results=1,
                                  include=["metadatas", "distances"])
        if not results["ids"][0]:
            return None
        similarity = 1.0 - results["distances"][0][0]  # cosine space: distance = 1 - similarity
        if self._debug:
            print(f"[cache] nearest similarity={similarity:.3f}")
        if similarity < threshold:
            return None
        return results["metadatas"][0][0]["answer"], similarity

    def store(self, question: str, answer: str) -> None:
        embedding = self._embedder.embed_query(question)
        if self._col.count() >= self._max_size:
            items = self._col.get(include=["metadatas"])
            oldest = min(zip(items["ids"], items["metadatas"]), key=lambda x: x[1].get("timestamp", 0.0))[0]
            self._col.delete(ids=[oldest])
        self._col.add(ids=[str(uuid.uuid4())], embeddings=[embedding],
                      metadatas=[{"answer": answer, "timestamp": time.time(), "question": question[:500]}])


# ─── Agent ───────────────────────────────────────────────────────────
class _AgentState(TypedDict):
    question: str                      # sanitized (already escaped) user question
    conversation_history: list[dict]
    clarification_history: list[str]   # sanitized Q/A pairs from clarify
    pending_queries: list[str]
    pending_paper_ids: list[str]
    pending_paper_search: dict
    executed_queries: list[str]
    chunks: dict[str, dict]            # chunk_id -> search result (best score kept)
    papers: dict[str, dict]            # paper_id -> catalog row from find_paper
    iterations: int
    mode: str
    next_action: str
    clarification_question: str
    answer: str
    citations: dict                    # citation check of the final answer (see _render_citations)
    done: bool


class PaperAgent:
    def __init__(self, hybrid: HybridRetriever, papers: PaperIndex, provenance: Provenance, *, debug: bool = True,
                 use_cache: bool = True, use_router: bool = True,
                 llm_model: str = config.LLM_MODEL, router_model: str = config.ROUTER_MODEL,
                 fallback_model: str = config.FALLBACK_MODEL, top_k: int = config.NUM_RETRIEVED,
                 max_iterations: int = config.MAX_ITERATIONS):
        self._hybrid, self._papers, self._prov = hybrid, papers, provenance
        self._llm_model, self._router_model, self._fallback_model = llm_model, router_model, fallback_model
        self._llm = make_chat_llm(llm_model)
        self._router_llm = make_chat_llm(router_model, config.ROUTER_REASONING_EFFORT)
        self._fallback_llm = make_chat_llm(fallback_model)
        self._top_k = top_k
        self._max_iterations = max_iterations
        self._debug = debug
        self._use_router = use_router
        self._cache = _SemanticCache(debug=debug) if use_cache else None
        self._token_usage: dict[str, dict[str, int]] = {}
        self.last_trace: dict = {}  # how the last question was answered (for evaluation)

        # Two graphs: the regular agent (mini) and the fallback agent (gpt-5.4).
        self._graph = self._build_graph(self._llm, llm_model)
        self._fallback_graph = self._build_graph(self._fallback_llm, fallback_model)

    def get_model_name(self) -> str:
        return f"llm={self._llm_model}, router={self._router_model}, fallback={self._fallback_model}"

    def _log(self, message: str) -> None:
        if self._debug:
            print(f"[agent] {message}")

    # ── token-usage tracking (from Lab 7.2) ──────────────────────────
    def _track_usage(self, response, model_name: str) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        bucket = self._token_usage.setdefault(model_name, {"input": 0, "output": 0, "cached": 0})
        bucket["input"] += usage.get("input_tokens", 0)
        bucket["output"] += usage.get("output_tokens", 0)
        bucket["cached"] += usage.get("input_token_details", {}).get("cache_read", 0)

    def get_token_usage(self) -> dict[str, dict[str, int]]:
        return {model: dict(counts) for model, counts in self._token_usage.items()}

    def print_token_usage(self) -> None:
        for model, counts in self._token_usage.items():
            print(f"{model}: {counts['input']} input, {counts['output']} output, {counts['cached']} cached")
        print(f"Approx. cost (USD): ${estimate_cost_usd(self._token_usage):.4f}")

    def reset_token_usage(self) -> None:
        self._token_usage.clear()

    def _invoke(self, llm: ChatOpenAI, model: str, system: str, content: str):
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
        self._track_usage(response, model)
        return response

    @staticmethod
    def _parse_json_response(response) -> dict:
        raw = (response.content if hasattr(response, "content") else str(response)).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return result if isinstance(result, dict) else {}

    # ── prompt pieces ────────────────────────────────────────────────
    def _format_chunk(self, r: dict, tag: str | None = None, preview: int | None = None) -> str:
        """A passage with its provenance as attributes; `tag` is the id the answer cites."""
        d = self._prov.describe(r["paper_id"], r["page_start"], r["page_end"], r["section"])
        pages = str(r["page_start"]) + (f"-{r['page_end']}" if r["page_end"] != r["page_start"] else "")
        text = r["text"] if preview is None else " ".join(r["text"].split())[:preview] + "..."
        attrs = {"id": tag or r["chunk_id"], "title": d["title"] or "", "authors": "; ".join(d["authors"]),
                 "year": d["year"] or "unknown", "source_type": d["kind"],
                 "in_volume": (d["container"] or {}).get("title") or "", "file": d["file"], "pdf_pages": pages,
                 "section": r["section"]}
        attr_str = " ".join(f'{k}="{escape_attr(v)}"' for k, v in attrs.items() if v != "")
        return f"<document {attr_str}>\n{escape_xml(text)}\n</document>"

    @staticmethod
    def _format_paper(p: dict, with_abstract: bool = True, tag: str | None = None) -> str:
        attrs = {"id": tag or p["paper_id"], "file": p["file"], "year": p.get("year") or "unknown",
                 "doc_type": p.get("doc_type"), "file_pages": p.get("pages")}
        attr_str = " ".join(f'{k}="{escape_attr(v)}"' for k, v in attrs.items())
        lines = [f"Title: {p.get('title') or '(unknown)'}", f"Authors: {', '.join(p.get('authors') or []) or '(unknown)'}",
                 f"Venue: {p.get('venue') or '(unknown)'}"]
        if with_abstract and p.get("abstract"):
            lines.append(f"Abstract: {p['abstract']}")
        return f"<paper {attr_str}>\n{escape_xml(chr(10).join(lines))}\n</paper>"

    @staticmethod
    def _format_conversation(history: list[dict]) -> str:
        # User turns were sanitized (and therefore escaped) on the way in; answers are escaped here.
        return "\n".join(f"{m['role'].capitalize()}: {m['content'] if m['role'] == 'user' else escape_xml(m['content'])}"
                         for m in history[-6:])

    def _best_chunks(self, chunks: dict[str, dict]) -> list[dict]:
        return sorted(chunks.values(), key=lambda r: r["score"], reverse=True)[:config.ANSWER_MAX_CHUNKS]

    @staticmethod
    def _tags(chunks: list[dict], papers: list[dict]) -> dict[str, tuple[str, dict]]:
        """Citation ids for one answer: S1.. for passages, P1.. for catalog records."""
        tags = {f"S{i}": ("chunk", r) for i, r in enumerate(chunks, 1)}
        tags.update({f"P{i}": ("paper", p) for i, p in enumerate(papers, 1)})
        return tags

    def _answer(self, llm, model: str, question: str, chunks: list[dict], papers: list[dict],
                conversation: list[dict], clarifications: list[str]) -> tuple[str, dict]:
        """Returns the model's raw answer (with [S1]-style ids) and the id -> source map."""
        tags = self._tags(chunks, papers)
        content = xml_prompt(
            ("papers", "\n".join(self._format_paper(p, tag=t) for t, (kind, p) in tags.items() if kind == "paper")),
            ("documents", "\n".join(self._format_chunk(r, tag=t) for t, (kind, r) in tags.items() if kind == "chunk")),
            ("conversation", self._format_conversation(conversation)),
            ("clarifications", "\n".join(clarifications)),
            ("user_question", question),
        )
        response = self._invoke(llm, model, ANSWER_SYSTEM, content)
        return (response.content if hasattr(response, "content") else str(response)), tags

    def _render_citations(self, raw: str, tags: dict[str, tuple[str, dict]]) -> tuple[str, dict]:
        """Replace the cited ids with numbers and append references built from catalog data only
        (fix C1). Ids that were not in the prompt are dropped and reported, never rendered."""
        if raw.strip() == "Exiting":
            return raw, {"cited": [], "invalid": [], "uncited": False}
        numbers: dict[str, int] = {}
        refs: list[str] = []
        invalid: list[str] = []

        def number(match: re.Match) -> str:
            out = []
            for tag in _split_tags(match.group(1)):
                if tag not in tags:
                    invalid.append(tag)
                    continue
                kind, src = tags[tag]
                key = src["chunk_id"] if kind == "chunk" else src["paper_id"]
                if key not in numbers:
                    numbers[key] = len(numbers) + 1
                    if kind == "chunk":
                        d = self._prov.describe(src["paper_id"], src["page_start"], src["page_end"], src["section"])
                    elif src.get("volume_id"):  # a paper inside a proceedings volume
                        d = self._prov.describe(src["volume_id"], src["page_start"], src["page_end"])
                    else:
                        d = self._prov.describe_paper(src["paper_id"])
                    refs.append(self._prov.reference(d))
                if numbers[key] not in out:
                    out.append(numbers[key])
            return "[" + ", ".join(map(str, out)) + "]" if out else ""

        text = re.sub(r"[ \t]+([.,;:])", r"\1", _TAG_GROUP_RE.sub(number, raw)).rstrip()  # "id [S9]." -> "id."
        if refs:
            text += "\n\nReferences\n" + "\n".join(f"[{i}] {ref}" for i, ref in enumerate(refs, 1))
        info = {"cited": list(numbers), "invalid": invalid, "uncited": not numbers}
        if invalid:
            self._log(f"dropped citation ids not in the prompt: {invalid}")
        return text, info

    # ── router, cache validation, satisfaction (from Lab 7.2) ────────
    def _route_query(self, question: str) -> str:
        """"direct" or "agent"; any parse failure takes the safe, more expensive "agent" path."""
        response = self._invoke(self._router_llm, self._router_model, ROUTER_SYSTEM,
                                xml_prompt(("user_question", question)))
        route = self._parse_json_response(response).get("route", "agent")
        route = route if route in {"direct", "agent"} else "agent"
        self._log(f"router -> {route!r}")
        return route

    def _validate_cached_answer(self, question: str, cached_answer: str) -> bool:
        response = self._invoke(self._router_llm, self._router_model, CACHE_VALIDATE_SYSTEM,
                                xml_prompt(("user_question", question), ("cached_answer", escape_xml(cached_answer))))
        valid = bool(self._parse_json_response(response).get("valid", False))
        self._log(f"cache validation -> {'VALID' if valid else 'INVALID'}")
        return valid

    def _check_answer_satisfactory(self, question: str, answer: str, chunks: list[dict]) -> bool:
        """The retrieved passages are included so the judge can check completeness. A "not found"
        answer escalates without asking the judge (fix B)."""
        if _NOT_FOUND_RE.search(answer):
            self._log("satisfaction -> ESCALATE (answer says the documents lack it)")
            return False
        response = self._invoke(self._router_llm, self._router_model, SATISFACTION_SYSTEM, xml_prompt(
            ("documents", "\n".join(self._format_chunk(r, tag=f"S{i}") for i, r in enumerate(chunks, 1))),
            ("user_question", question), ("answer", escape_xml(answer))))
        satisfactory = bool(self._parse_json_response(response).get("satisfactory", False))
        self._log(f"satisfaction -> {'OK' if satisfactory else 'ESCALATE'}")
        return satisfactory

    def _direct_retrieve_and_answer(self, question: str, conversation: list[dict]) -> tuple[str, bool, list[dict], dict]:
        """One hybrid retrieval + one answer call. Returns (answer, satisfactory, chunks, citation check)."""
        chunks = self._hybrid.search(question, k=self._top_k)
        raw, tags = self._answer(self._llm, self._llm_model, question, chunks, [], conversation, [])
        ok = self._check_answer_satisfactory(question, raw, chunks)
        answer, citations = self._render_citations(raw, tags)
        return answer, ok, chunks, citations

    # ── the LangGraph agent ──────────────────────────────────────────
    def _build_graph(self, llm: ChatOpenAI, model: str):
        def retrieve_node(state: _AgentState) -> dict:
            chunks = dict(state["chunks"])
            executed = list(state["executed_queries"])
            rows = [row for row in map(self._papers.get, state["pending_paper_ids"]) if row]
            paper_ids = [row["paper_id"] for row in rows if not row.get("volume_id")] or None
            page_ranges: dict[str, list[tuple[int, int]]] = {}
            for row in rows:  # a paper inside a volume is searched by its pages
                if row.get("volume_id"):
                    page_ranges.setdefault(row["volume_id"], []).append((row["page_start"], row["page_end"]))
            for query in state["pending_queries"]:
                scope = f" [in {len(rows)} paper(s)]" if rows else ""
                self._log(f"retrieving for: {query!r}{scope}")
                for r in self._hybrid.search(query, k=self._top_k, paper_ids=paper_ids, page_ranges=page_ranges):
                    if r["chunk_id"] not in chunks or r["score"] > chunks[r["chunk_id"]]["score"]:
                        chunks[r["chunk_id"]] = r
                executed.append(query + scope)
            return {"chunks": chunks, "executed_queries": executed, "pending_queries": [],
                    "pending_paper_ids": [], "iterations": state["iterations"] + 1}

        def find_paper_node(state: _AgentState) -> dict:
            req = state["pending_paper_search"]
            rows = self._papers.find(query=req.get("paper_query") or None, author=req.get("author") or None,
                                     year_from=_as_int(req.get("year_from")), year_to=_as_int(req.get("year_to")),
                                     k=config.FIND_PAPER_K)
            self._log(f"find_paper {req}: {len(rows)} paper(s)")
            papers = dict(state["papers"])
            papers.update({r["paper_id"]: r for r in rows})
            tag = "FIND_PAPER " + json.dumps(req, ensure_ascii=False) + f" -> {len(rows)} paper(s)"
            return {"papers": papers, "executed_queries": list(state["executed_queries"]) + [tag],
                    "pending_paper_search": {}, "iterations": state["iterations"] + 1}

        def plan_node(state: _AgentState) -> dict:
            """The agent's brain: pick the next action as a JSON object."""
            idle = {"pending_queries": [], "pending_paper_ids": [], "pending_paper_search": {},
                    "clarification_question": ""}
            if state["iterations"] >= self._max_iterations:
                self._log("max iterations — forcing answer.")
                return {**idle, "next_action": "answer"}
            content = xml_prompt(
                ("conversation", self._format_conversation(state["conversation_history"])),
                ("clarifications", "\n".join(state["clarification_history"])),
                ("executed_steps", escape_xml("\n".join(f"- {q}" for q in state["executed_queries"])) or "(none)"),
                ("papers", "\n".join(self._format_paper(p, with_abstract=False) for p in state["papers"].values())),
                ("passage_previews", "\n".join(self._format_chunk(r, preview=config.PLAN_PREVIEW_CHARS)
                                               for r in self._best_chunks(state["chunks"]))),
                ("user_question", state["question"]),
            )
            result = self._parse_json_response(self._invoke(llm, model, PLAN_SYSTEM, content))
            # With nothing retrieved yet, an unparseable plan searches the question itself
            # instead of answering from no evidence.
            default = "retrieve" if not state["chunks"] and not state["papers"] else "answer"
            action = result.get("action", default)
            if action == "answer":  # a planner's "answer" with no evidence still searches once
                action = default
            self._log(f"action={action} reasoning={result.get('reasoning', '')!r}")
            if action == "retrieve":
                queries = [q for q in result.get("queries", []) if isinstance(q, str) and q.strip()]
                return {**idle, "next_action": "retrieve", "pending_queries": queries or [state["question"]],
                        "pending_paper_ids": [p for p in result.get("paper_ids", []) if isinstance(p, str)]}
            if action == "find_paper":
                search = {k: result.get(k) for k in ("paper_query", "author", "year_from", "year_to") if result.get(k)}
                return {**idle, "next_action": "find_paper", "pending_paper_search": search}
            if action == "clarify":
                return {**idle, "next_action": "clarify",
                        "clarification_question": result.get("clarification") or "Could you clarify your question?"}
            return {**idle, "next_action": "answer"}

        def clarify_node(state: _AgentState) -> dict:
            # Batch mode has no user to ask, so it records that and re-plans with what it has.
            question = state["clarification_question"]
            history = list(state["clarification_history"])
            if state["mode"] == "chat":
                print(f"\nAssistant: {question}")
                reply = sanitize_user_text(input("You: ").strip())
                history.append(f"Q: {escape_xml(question)}\nA: {reply}")
            else:
                history.append(f"Q: {escape_xml(question)}\n[no clarification available in batch mode]")
            return {"clarification_history": history, "clarification_question": "",
                    "iterations": state["iterations"] + 1}

        def answer_node(state: _AgentState) -> dict:
            raw, tags = self._answer(llm, model, state["question"], self._best_chunks(state["chunks"]),
                                     list(state["papers"].values()), state["conversation_history"],
                                     state["clarification_history"])
            answer, citations = self._render_citations(raw, tags)
            return {"answer": answer, "citations": citations, "done": True}

        def route_from_plan(state: _AgentState) -> str:
            action = state.get("next_action", "answer")
            if action == "retrieve" and state["pending_queries"]:
                return "retrieve"
            if action == "find_paper" and state["pending_paper_search"]:
                return "find_paper"
            if action == "clarify":
                return "clarify"
            return "answer"

        graph = StateGraph(_AgentState)
        graph.add_node("plan", plan_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("find_paper", find_paper_node)
        graph.add_node("clarify", clarify_node)
        graph.add_node("answer", answer_node)
        graph.set_entry_point("plan")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("find_paper", "plan")
        graph.add_edge("clarify", "plan")
        graph.add_conditional_edges("plan", route_from_plan, {
            "retrieve": "retrieve", "find_paper": "find_paper", "clarify": "clarify", "answer": "answer",
        })
        graph.add_edge("answer", END)
        return graph.compile()

    def _run_agent(self, question: str, mode: str, conversation: list[dict], use_fallback: bool = False) -> str:
        graph = self._fallback_graph if use_fallback else self._graph
        initial: _AgentState = {
            "question": question, "conversation_history": conversation, "clarification_history": [],
            "pending_queries": [], "pending_paper_ids": [], "pending_paper_search": {},
            "executed_queries": [], "chunks": {}, "papers": {}, "iterations": 0, "mode": mode,
            "next_action": "plan", "clarification_question": "", "answer": "", "citations": {}, "done": False,
        }
        final = graph.invoke(initial, {"recursion_limit": 4 * self._max_iterations + 10})
        self.last_trace.update({"steps": final["executed_queries"], "citations": final["citations"],
                                "chunks": [r["chunk_id"] for r in self._best_chunks(final["chunks"])],
                                "papers": list(final["papers"])})
        return final["answer"]

    def _run(self, raw_question: str, mode: str, conversation: Optional[list[dict]] = None) -> str:
        """Sanitize -> cache -> route -> (direct | full agent); every answer is stored in the cache."""
        question = sanitize_user_text(raw_question)  # before the cache, the retriever and any prompt
        sanitized = was_sanitized(raw_question, question)
        if sanitized:
            self._log(f"sanitized question: {question!r}")
        conversation = conversation or []
        self.last_trace = {"question": question, "path": None}

        # 1. Semantic cache: a validated hit is by far the cheapest outcome.
        if self._cache:
            hit = self._cache.lookup(question, config.CACHE_SIMILARITY_THRESHOLD)
            if hit and self._validate_cached_answer(question, hit[0]):
                self._log("returning validated cached answer")
                self.last_trace.update({"path": "cache", "cache_similarity": round(hit[1], 3)})
                return hit[0]

        # 2. Route. With routing off we always run the full agent (the un-optimized baseline).
        #    A sanitized question also skips the direct path (fix A): leftover attack words would
        #    steer a search of the raw text, while the planner writes clean search queries.
        if sanitized:
            self._log("sanitizer removed content — skipping the direct path")
            route = "agent"
        else:
            route = self._route_query(question) if self._use_router else "agent"

        # 3. Direct path: one retrieval + one answer; escalate to the gpt-5.4 agent if unsatisfactory.
        if route == "direct":
            answer, ok, chunks, citations = self._direct_retrieve_and_answer(question, conversation)
            self.last_trace.update({"path": "direct", "chunks": [r["chunk_id"] for r in chunks], "citations": citations,
                                    "satisfaction": ok})
            if not ok:
                self._log("direct answer unsatisfactory — escalating to the fallback agent")
                answer = self._run_agent(question, mode, conversation, use_fallback=True)
                self.last_trace["path"] = "direct->fallback"
        else:
            answer = self._run_agent(question, mode, conversation)
            self.last_trace["path"] = "agent"
        if self._cache:
            self._cache.store(question, answer)
        return answer

    def query(self, question: str) -> str:
        """Batch mode — no clarification prompts (for evaluation and one-off questions)."""
        return self._run(question, "batch")

    def chat(self) -> None:
        print("Research Paper Navigator. Ask about the papers; 'exit' or 'quit' to stop.\n")
        conversation: list[dict] = []
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not user_input:
                continue
            try:
                answer = self._run(user_input, "chat", conversation)
            except Exception as e:  # one failed call (rate limit, timeout) must not end the session
                print(f"Error: {type(e).__name__}: {e}")
                continue
            print(f"\nAssistant: {answer}\n")
            if answer.strip() == "Exiting":
                print("Goodbye.")
                break
            conversation.append({"role": "user", "content": sanitize_user_text(user_input)})
            conversation.append({"role": "assistant", "content": answer})


def build_agent(tag: str | None = None, **kwargs) -> PaperAgent:
    """Load the index once and wire up the agent (kwargs go to PaperAgent)."""
    return PaperAgent(HybridRetriever(tag), PaperIndex(), Provenance(), **kwargs)

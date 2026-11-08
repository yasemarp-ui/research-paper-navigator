"""Correct provenance for citations: checked paper metadata, papers inside proceedings volumes,
manual overrides, and the reference text the agent attaches to its answers.

The agent cites passages by tag ([S1]); this module turns a tag's chunk into a reference built
only from catalog data — title, authors, year, file and the chunk's own PDF pages — so the
model never writes a title, an author or a page number itself.

  catalog/metadata_check.json      second, independent extraction per paper (gpt-5.4-mini),
                                   cached by sha256; disagreements are flagged for review
  catalog/volume_papers.json       papers found inside multi-paper volumes, by first page
  catalog/metadata_overrides.json  manual corrections; they always win (meant to be committed)
  catalog/metadata_review.csv      every title/author a citation can show, for a human check

`paper_navigator.py provenance` makes the API calls (cached) and applies everything to
catalog/papers.json; the `catalog` command reapplies the cached results without API calls.
"""
from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import config
from catalog import PAGE_SEPARATOR, _now, _read_json, _write_json
from prompts import METADATA_CHECK_SYSTEM, VOLUME_PAGE_SYSTEM
from safety import escape_attr, escape_xml

# ─── Deterministic repairs ───────────────────────────────────────────
# Spacing accents that LaTeX-made PDFs print next to the letter instead of on it. After NFKC
# (catalog.clean_page_text) most become " " + a combining mark, e.g. "Hern ́andez".
_SPACING_ACCENTS = {"`": "̀", "´": "́", "ˆ": "̂", "˜": "̃", "¨": "̈",
                    "˘": "̆", "ˇ": "̌", "˚": "̊", "¸": "̧"}
_ACCENT_RE = re.compile(r"(?<=[A-Za-z]) ?([`´ˆ˜¨˘ˇ˚¸̀-ͯ]) ?([A-Za-z])")
_SPLIT_ACCENTED_RE = re.compile(r"\b([A-Z]) ([À-ſ])")   # "W üstholz" -> "Wüstholz"
_MARKERS_RE = re.compile(r"[\s\d⋆∗*†‡§¶,()]+$")


def fix_accents(text: str) -> str:
    def attach(m: re.Match) -> str:
        mark = _SPACING_ACCENTS.get(m.group(1), m.group(1))
        return m.group(2) + mark

    text = _ACCENT_RE.sub(attach, text)
    text = _SPLIT_ACCENTED_RE.sub(r"\1\2", text)
    return unicodedata.normalize("NFC", text)


def clean_title(title: str | None) -> str | None:
    if not title:
        return None
    title = re.sub(r"\s+([’']s)\b", r"\1", fix_accents(title))  # "Dijkstra ’s" -> "Dijkstra’s"
    return " ".join(title.split()) or None


def clean_author(name: str) -> str:
    return _MARKERS_RE.sub("", " ".join(fix_accents(name).split()))


def _norm(text: str | None) -> str:
    """Comparison key: accents, case, spacing and punctuation removed."""
    text = unicodedata.normalize("NFKD", text or "")
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in text if not unicodedata.combining(c)).lower())


def _surnames(authors: list[str]) -> set[str]:
    return {_norm(a.split()[-1]) for a in authors if a.split()}


# ─── Model calls ─────────────────────────────────────────────────────
def _make_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.PROVENANCE_MODEL, api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url=config.OPENROUTER_BASE_URL, timeout=120, max_retries=2,
                      reasoning_effort=config.PROVENANCE_REASONING_EFFORT)


def _ask_json(llm, system: str, content: str) -> tuple[dict, dict]:
    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
    usage = getattr(response, "usage_metadata", None) or {}
    raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        parsed = {}
    return (parsed if isinstance(parsed, dict) else {},
            {"input": int(usage.get("input_tokens", 0)), "output": int(usage.get("output_tokens", 0))})


def _volume_candidates(pages: list[str]) -> list[int]:
    """1-based pages that may start a paper: 'Abstract' near the top of the page."""
    return [i + 1 for i, text in enumerate(pages) if re.search(r"\bAbstract\b", text[: config.VOLUME_PAGE_CHARS])]


def _back_matter_start(pages: list[str]) -> int | None:
    """First page of a closing author index, which ends the last paper of a volume."""
    for i in range(len(pages) - 1, max(-1, len(pages) - 15), -1):
        if re.match(r"\s*Author Index\b", pages[i]):
            return i + 1
    return None


def build_provenance(use_llm: bool = True) -> dict:
    """Run (or reuse) the metadata check and the volume scan, then apply and write the review."""
    catalog = _read_json(config.PAPERS_FILE)
    if not catalog:
        raise SystemExit("No catalog/papers.json yet — run the 'catalog' command first.")
    checks = _read_json(config.METADATA_CHECK_FILE, {}) or {}
    volumes = _read_json(config.VOLUME_PAPERS_FILE, {}) or {}
    llm = _make_llm() if use_llm else None
    usage = {"input": 0, "output": 0, "calls": 0}

    def fresh(entry: dict | None, sha: str) -> bool:
        return bool(entry and entry.get("sha256") == sha and entry.get("version") == config.PROVENANCE_VERSION)

    jobs = []  # (kind, key, content)
    page_text = {}
    for p in catalog["papers"]:
        pid, sha = p["paper_id"], p["sha256"]
        pages = (_read_json(config.PAGES_DIR / f"{pid}.json") or {}).get("pages", [])
        page_text[pid] = pages
        if not fresh(checks.get(pid), sha):
            first = PAGE_SEPARATOR.join(pages)[: config.PROVENANCE_MAX_CHARS]
            jobs.append(("paper", pid, f'<document paper_id="{escape_attr(pid)}">\n{escape_xml(first)}\n</document>'))
        if p["pages"] >= config.VOLUME_MIN_PAGES and not fresh(volumes.get(pid), sha):
            for page_no in _volume_candidates(pages):
                text = pages[page_no - 1][: config.VOLUME_PAGE_CHARS]
                jobs.append(("page", (pid, page_no), f'<page number="{page_no}">\n{escape_xml(text)}\n</page>'))

    if jobs and not llm:
        print(f"[provenance] {len(jobs)} check(s) need the model; skipped (--no-llm).")
    elif jobs:
        print(f"Provenance model {config.PROVENANCE_MODEL}: {len(jobs)} call(s), {config.METADATA_WORKERS} in parallel...")

        def run(job):
            kind, _, content = job
            try:
                return _ask_json(llm, METADATA_CHECK_SYSTEM if kind == "paper" else VOLUME_PAGE_SYSTEM, content)
            except Exception as e:  # one failed call is retried on the next run
                return {"_error": type(e).__name__}, {"input": 0, "output": 0}

        page_results: dict[str, dict[int, dict]] = {}
        with ThreadPoolExecutor(max_workers=config.METADATA_WORKERS) as pool:
            for (kind, key, _), (result, u) in zip(jobs, pool.map(run, jobs)):
                usage["input"] += u["input"]
                usage["output"] += u["output"]
                usage["calls"] += 1
                if kind == "paper":
                    if "_error" not in result and result:
                        sha = next(p["sha256"] for p in catalog["papers"] if p["paper_id"] == key)
                        checks[key] = {"sha256": sha, "version": config.PROVENANCE_VERSION, "raw": result}
                else:
                    page_results.setdefault(key[0], {})[key[1]] = result

        for p in catalog["papers"]:
            pid = p["paper_id"]
            if pid not in page_results:
                continue
            if any("_error" in r for r in page_results[pid].values()):
                print(f"[provenance] {pid}: some page checks failed; rerun to retry this volume.")
                continue
            starts = [{"start_page": pg, "title": r.get("title"), "authors": r.get("authors") or []}
                      for pg, r in sorted(page_results[pid].items()) if r.get("is_paper_start")]
            volumes[pid] = {"sha256": p["sha256"], "version": config.PROVENANCE_VERSION, "pages": p["pages"],
                            "back_matter_start": _back_matter_start(page_text[pid]), "papers": starts}
        _write_json(config.METADATA_CHECK_FILE, checks)
        _write_json(config.VOLUME_PAPERS_FILE, volumes)

    apply_to_catalog(catalog)
    _write_json(config.PAPERS_FILE, catalog)
    flagged = write_review(catalog)
    price = config.APPROX_PRICE_PER_MTOK.get(config.PROVENANCE_MODEL, {"input": 0, "output": 0})
    cost = usage["input"] / 1e6 * price["input"] + usage["output"] / 1e6 * price["output"]
    n_inner = sum(len(v["papers"]) for v in effective_volumes(catalog).values())
    print(f"\nProvenance: {len(catalog['papers'])} papers, {n_inner} papers inside "
          f"{len(effective_volumes(catalog))} volume(s); {flagged} review row(s) flagged.\n"
          f"Calls this run: {usage['calls']} (~${cost:.2f}). Review: {config.REVIEW_FILE.relative_to(config.BASE_DIR)}; "
          f"corrections: {config.OVERRIDES_FILE.relative_to(config.BASE_DIR)}.\n"
          "Then run 'index' so the paper vectors pick up corrected titles (only changed papers are re-embedded).")
    return usage


# ─── Overrides and effective metadata ────────────────────────────────
_OVERRIDES_TEMPLATE = {
    "_how_to": [
        "Manual corrections. They always win over the extracted metadata.",
        "papers: {paper_id: {title, authors, year, venue, doc_type}} - give only the fields to change.",
        "volume_papers: {volume_paper_id: [{start_page, title, authors}]} - adds or corrects the paper "
        "starting on that PDF page; {start_page, remove: true} drops a wrongly detected one.",
        "After editing, run: paper_navigator.py provenance --no-llm, then index.",
    ],
    "papers": {},
    "volume_papers": {},
}


def load_overrides() -> dict:
    data = _read_json(config.OVERRIDES_FILE)
    if data is None:
        _write_json(config.OVERRIDES_FILE, _OVERRIDES_TEMPLATE)
        data = _OVERRIDES_TEMPLATE
    return data


def apply_to_catalog(catalog: dict) -> None:
    """Set each paper's `metadata` to the corrected version. The extraction-only result is kept
    in `metadata_auto`, so this can run any number of times."""
    checks = _read_json(config.METADATA_CHECK_FILE, {}) or {}
    volumes = _read_json(config.VOLUME_PAPERS_FILE, {}) or {}
    overrides = load_overrides().get("papers", {})
    for p in catalog["papers"]:
        auto = p.setdefault("metadata_auto", p["metadata"])
        p.pop("content_mismatch", None)
        meta, src, flags = dict(auto), dict(p.get("metadata_source", {})), []
        check = checks.get(p["paper_id"], {})
        raw = check.get("raw", {}) if check.get("sha256") == p["sha256"] else {}

        title = clean_title(raw.get("title"))
        if title:
            if auto.get("title") and _norm(title) != _norm(auto["title"]):
                flags.append("title differs between extractions")
            meta["title"], src["title"] = title, "check"
        else:
            meta["title"] = clean_title(auto.get("title"))
        authors = [clean_author(a) for a in raw.get("authors") or [] if isinstance(a, str) and a.strip()]
        if authors:
            if auto.get("authors") and _surnames(authors) != _surnames(auto["authors"]):
                flags.append("authors differ between extractions")
            meta["authors"], src["authors"] = authors, "check"
        else:
            meta["authors"] = [clean_author(a) for a in auto.get("authors") or []]
        if isinstance(raw.get("year"), int) and meta.get("year") and abs(raw["year"] - meta["year"]) > 1:
            if abs(raw["year"] - meta["year"]) >= config.WRONG_FILE_YEAR_GAP:
                # Not a revision date: the file holds a different work than its DBLP name says,
                # so a citation must use what the document itself prints.
                flags.append(f"file is a different work than its name: name says {meta['year']}, document {raw['year']}")
                meta["year"], src["year"] = raw["year"], "check"
                p["content_mismatch"] = True
            else:
                flags.append(f"year: file name {meta['year']}, document {raw['year']}")
        if not meta.get("year") and isinstance(raw.get("year"), int):
            meta["year"], src["year"] = raw["year"], "check"
        if p.get("content_mismatch") and src.get("authors") == "filename":
            meta["authors"], meta["first_author"], src["authors"] = [], None, None  # the name's author is wrong here
        if raw.get("venue"):
            meta["venue"], src["venue"] = " ".join(str(raw["venue"]).split()), "check"
        if volumes.get(p["paper_id"], {}).get("papers"):
            meta["doc_type"], src["doc_type"] = "proceedings_volume", "volume_scan"
        elif raw.get("doc_type") in {"proceedings_volume", "front_matter", "rebuttal", "preface"}:
            meta["doc_type"], src["doc_type"] = raw["doc_type"], "check"
        if not meta.get("title") and meta.get("doc_type") != "rebuttal":
            flags.append("no title")
        if not meta.get("authors") and meta.get("doc_type") in {"paper", "preprint"}:
            flags.append("no authors")
        if not raw:
            flags.append("not checked by the second model")

        for field, value in (overrides.get(p["paper_id"]) or {}).items():
            meta[field], src[field] = value, "override"
        if p["paper_id"] in overrides:
            flags = []  # a human has looked at this paper
        if meta.get("authors") and src.get("authors") in {"check", "override"}:
            # From the document, not the file name: a mislabeled file must not match its name's author.
            meta["first_author"] = meta["authors"][0].split()[-1]
        p["metadata"], p["metadata_source"], p["review_flags"] = meta, src, flags


def effective_volumes(catalog: dict | None = None) -> dict[str, dict]:
    """Papers inside each volume with overrides applied and page ranges filled in:
    {volume_id: {"pages": n, "papers": [{start_page, end_page, title, authors}]}}."""
    volumes = _read_json(config.VOLUME_PAPERS_FILE, {}) or {}
    overrides = load_overrides().get("volume_papers", {})
    out = {}
    for vid in sorted(set(volumes) | set(overrides)):
        vol = volumes.get(vid, {})
        by_start = {e["start_page"]: dict(e) for e in vol.get("papers", [])}
        for e in overrides.get(vid, []):
            if e.get("remove"):
                by_start.pop(e["start_page"], None)
            else:
                by_start[e["start_page"]] = {**by_start.get(e["start_page"], {}), **e, "override": True}
        n_pages = vol.get("pages") or 0
        last = (vol.get("back_matter_start") or n_pages + 1) - 1
        starts = sorted(by_start)
        papers = []
        for i, start in enumerate(starts):
            e = by_start[start]
            end = starts[i + 1] - 1 if i + 1 < len(starts) else last
            papers.append({"start_page": start, "end_page": max(start, end), "title": clean_title(e.get("title")),
                           "authors": [clean_author(a) for a in e.get("authors") or []],
                           "override": bool(e.get("override"))})
        if papers:
            out[vid] = {"pages": n_pages, "papers": papers}
    return out


def write_review(catalog: dict) -> int:
    """Write metadata_review.csv; returns how many rows carry a flag."""
    rows, flagged = [], 0
    for p in catalog["papers"]:
        m = p["metadata"]
        flags = p.get("review_flags", [])
        flagged += bool(flags)
        rows.append({"kind": "document", "id": p["paper_id"], "file": p["file"], "pdf_pages": f"1-{p['pages']}",
                     "doc_type": m.get("doc_type"), "year": m.get("year"), "title": m.get("title"),
                     "authors": "; ".join(m.get("authors") or []), "venue": m.get("venue"),
                     "title_source": p.get("metadata_source", {}).get("title"), "check": " | ".join(flags)})
    files = {p["paper_id"]: p for p in catalog["papers"]}
    for vid, vol in effective_volumes(catalog).items():
        vmeta = files[vid]["metadata"] if vid in files else {}
        for e in vol["papers"]:
            flags = [] if e["override"] else ["detected automatically"]
            if not e["title"] or not e["authors"]:
                flags.append("missing title or authors")
            flagged += bool(flags) and not e["override"]
            rows.append({"kind": "paper in volume", "id": f"{vid}@p{e['start_page']}", "file": files.get(vid, {}).get("file", vid),
                         "pdf_pages": f"{e['start_page']}-{e['end_page']}", "doc_type": "paper",
                         "year": vmeta.get("year"), "title": e["title"], "authors": "; ".join(e["authors"]),
                         "venue": vmeta.get("title"), "title_source": "override" if e["override"] else "volume_scan",
                         "check": " | ".join(flags)})
    config.REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.REVIEW_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return flagged


# ─── Citation lookup (used by the agent) ─────────────────────────────
def _authors_text(authors: list[str], editors: bool = False) -> str:
    if not authors:
        return "Unknown authors"
    if len(authors) > 6:
        text = ", ".join(authors[:6]) + ", et al."
    elif len(authors) == 1:
        text = authors[0]
    else:
        text = ", ".join(authors[:-1]) + " and " + authors[-1]
    return text + (" (eds.)" if editors else "")


def _sentence(text: str) -> str:
    """End with a period unless the text already ends a sentence ("...Code Generation?")."""
    return text if text.endswith((".", "?", "!")) else text + "."


def _clean_section(section: str) -> str:
    """The guessed section heading, or "" when it does not look like a real heading (the guess
    in chunking.find_headings sometimes picks up a formula line, e.g. "16 ESYNÐ ESYN.append")."""
    if not re.fullmatch(r"[A-Za-z0-9 .:\-–,'’()&/]{3,80}", section or ""):
        return ""
    letters = sum(c.isalpha() for c in section)
    return section if letters >= 0.6 * len(section.replace(" ", "")) else ""


def _page_text(start: int, end: int) -> str:
    return f"PDF p. {start}" if start == end else f"PDF pp. {start}–{end}"


class Provenance:
    """Who wrote the text in a chunk, and how to cite it."""

    def __init__(self):
        catalog = _read_json(config.PAPERS_FILE)
        if not catalog:
            raise SystemExit("No catalog/papers.json yet — run the 'catalog' command first.")
        self._papers = {p["paper_id"]: p for p in catalog["papers"]}
        self._volumes = effective_volumes(catalog)

    def describe(self, paper_id: str, page_start: int, page_end: int, section: str = "") -> dict:
        """Provenance of a passage: the paper (or the paper inside a volume) it comes from."""
        p = self._papers.get(paper_id, {"file": paper_id, "metadata": {}})
        m = p["metadata"]
        desc = {"kind": m.get("doc_type") or "paper", "title": m.get("title"), "authors": m.get("authors") or [],
                "year": m.get("year"), "venue": m.get("venue"), "file": p["file"], "page_start": page_start,
                "page_end": page_end, "section": section, "container": None}
        vol = self._volumes.get(paper_id)
        if vol:
            container = {"title": m.get("title"), "editors": m.get("authors") or []}
            inner = [e for e in vol["papers"] if e["start_page"] <= page_start <= e["end_page"]]
            if inner:
                e = inner[0]
                desc.update(kind="paper_in_volume", title=e["title"], authors=e["authors"], container=container)
            else:
                desc.update(kind="volume_front_or_back", container=container)
        return desc

    def describe_paper(self, paper_id: str) -> dict:
        p = self._papers.get(paper_id, {"file": paper_id, "metadata": {}, "pages": 0})
        m = p["metadata"]
        return {"kind": m.get("doc_type") or "paper", "title": m.get("title"), "authors": m.get("authors") or [],
                "year": m.get("year"), "venue": m.get("venue"), "file": p["file"], "page_start": None,
                "page_end": None, "section": "", "container": None}

    @staticmethod
    def reference(d: dict) -> str:
        """One reference line, built only from catalog data."""
        year = f" ({d['year']})" if d.get("year") else ""
        pages = f", {_page_text(d['page_start'], d['page_end'])}" if d.get("page_start") else ""
        section = f' — section "{_clean_section(d["section"])}"' if _clean_section(d.get("section", "")) else ""
        title = _sentence(d.get("title") or "Untitled")
        if d["kind"] == "paper_in_volume":
            c = d["container"]
            return (f"{title} {_authors_text(d['authors'])}. In: {c['title'] or 'proceedings volume'}"
                    f"{year}. {d['file']}{pages}{section}.")
        if d["kind"] == "volume_front_or_back":
            return f"Front or back matter of {d['container']['title'] or 'a proceedings volume'}{year}. {d['file']}{pages}{section}."
        if d["kind"] in {"proceedings_volume", "front_matter"}:
            return f"{title} {_authors_text(d['authors'], editors=True)}{year}. {d['file']}{pages}{section}."
        if d["kind"] == "rebuttal":
            return f"Author response to reviewers{': ' + d['title'] if d.get('title') else ''}{year}. {d['file']}{pages}."
        venue = f" {d['venue']}." if d.get("venue") else ""
        return f"{title} {_authors_text(d['authors'])}{year}.{venue} {d['file']}{pages}{section}."

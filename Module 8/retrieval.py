"""Index and search: chunk and paper embeddings in Chroma, plus BM25, fused into one ranking.

Two Chroma collections live in config.CHROMA_DIR:
  chunks_<tag>   one vector per chunk (id = chunk_id) for passage retrieval
  papers         one vector per paper (title, authors, abstract) for finding the right paper

tokenize and the 50/50 BM25/vector weighting come from the Lab 7.2 solution. Unlike the lab,
results are fused by chunk_id rather than by file name, so passages from one paper stay
separate; the two rankings are combined by reciprocal rank fusion instead of min-max score
normalization (see HybridRetriever.search); and embedding runs in batches that skip ids already stored, so an interrupted run
resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

import chromadb
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

import config
from provenance import effective_volumes
from catalog import _now, _read_json, _write_json
from chunking import strategy_tag

INDEX_MANIFEST = config.CHROMA_DIR / "index_manifest.json"

# ─── Shared helpers (from Lab 7.2) ───────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=config.OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,  # OpenRouter needs raw text, not pre-tokenized input
    )


# ─── Catalog access ──────────────────────────────────────────────────
def default_tag() -> str:
    return strategy_tag(**config.CHUNKING)


def chunk_collection_name(tag: str) -> str:
    return f"chunks_{tag}"


def load_manifest(tag: str) -> dict:
    manifest = _read_json(config.CATALOG_DIR / tag / "manifest.json")
    if not manifest:
        raise SystemExit(f"No chunk catalog for '{tag}' — run the 'chunk' (or 'catalog') command first.")
    return manifest


def load_papers() -> list[dict]:
    """Included papers from catalog/papers.json."""
    catalog = _read_json(config.PAPERS_FILE)
    if not catalog:
        raise SystemExit("No catalog/papers.json yet — run the 'catalog' command first.")
    return [p for p in catalog["papers"] if p["include"]]


def load_chunks(tag: str, papers: list[dict] | None = None) -> list[dict]:
    """Chunk records of the included papers, in catalog order."""
    chunks = []
    for paper in papers if papers is not None else load_papers():
        rec = _read_json(config.CATALOG_DIR / tag / "chunks" / f"{paper['paper_id']}.json")
        if rec is None:
            raise SystemExit(f"Chunks for {paper['paper_id']} are missing — rerun the 'chunk' command.")
        chunks.extend(rec["chunks"])
    return chunks


def _year(meta: dict) -> int:
    return meta.get("year") or -1  # Chroma metadata cannot hold None


def _chunk_metadata(chunk: dict, meta: dict) -> dict:
    return {"paper_id": chunk["paper_id"], "seq": chunk["seq"], "page_start": chunk["page_start"],
            "page_end": chunk["page_end"], "section": chunk["section"], "year": _year(meta),
            "doc_type": meta.get("doc_type") or "other"}


def _paper_text(paper: dict) -> str:
    """What the paper-level vector represents: title, authors and abstract (or, when the
    abstract is missing, the start of page 1)."""
    meta = paper["metadata"]
    summary = meta.get("abstract")
    if not summary:
        pages = _read_json(config.PAGES_DIR / f"{paper['paper_id']}.json") or {}
        summary = (pages.get("pages") or [""])[0]
    parts = [meta.get("title") or paper["paper_id"], "; ".join(meta.get("authors") or []),
             summary[: config.PAPER_SUMMARY_MAX_CHARS]]
    return "\n".join(p for p in parts if p)


def _paper_metadata(paper: dict, text: str) -> dict:
    meta = paper["metadata"]
    return {"paper_id": paper["paper_id"], "file": paper["file"],
            "title": meta.get("title") or "", "authors": "; ".join(meta.get("authors") or []),
            "first_author": meta.get("first_author") or "", "year": _year(meta),
            "venue": meta.get("venue") or "", "doc_type": meta.get("doc_type") or "other",
            "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16]}


# ─── Chroma ──────────────────────────────────────────────────────────
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                     settings=chromadb.Settings(anonymized_telemetry=False))


def _open_collection(client: chromadb.ClientAPI, name: str, metadata: dict, create: bool = True):
    try:
        return client.get_collection(name)
    except Exception:
        if not create:
            raise SystemExit(f"No '{name}' index yet — run the 'index' command first.")
        return client.create_collection(name, metadata={"hnsw:space": "cosine", **metadata})


def _drop_collection(client: chromadb.ClientAPI, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass  # nothing to drop


def _check_fresh(collection, manifest: dict) -> None:
    stored = (collection.metadata or {}).get("config_hash")
    if collection.count() and stored != manifest["config_hash"]:
        raise SystemExit(f"The '{collection.name}' index was built from different chunks "
                         f"(config hash {stored}, catalog now {manifest['config_hash']}). "
                         "Rerun 'index --rebuild'.")


def _embed_into(collection, emb: OpenAIEmbeddings, items: list[tuple[str, str, dict]], label: str) -> None:
    """Embed (id, text, metadata) items in batches; each batch is stored as soon as it returns."""
    batch_size = config.EMBED_BATCH
    started = time.time()
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        for attempt in range(3):
            try:
                vectors = emb.embed_documents([text for _, text, _ in batch])
                break
            except Exception as e:  # transient network/rate-limit errors: back off and retry
                if attempt == 2:
                    raise
                print(f"  [{label}] batch {i // batch_size + 1} failed ({e}); retrying...")
                time.sleep(5 * (attempt + 1))
        collection.add(ids=[b[0] for b in batch], embeddings=vectors,
                       documents=[b[1] for b in batch], metadatas=[b[2] for b in batch])
        print(f"  [{label}] {min(i + batch_size, len(items))}/{len(items)} "
              f"({time.time() - started:.0f}s)")


def build_index(tag: str | None = None, rebuild: bool = False) -> dict:
    """Embed every included chunk and paper that is not stored yet. Safe to rerun."""
    tag = tag or default_tag()
    manifest = load_manifest(tag)
    papers = load_papers()
    meta_by_paper = {p["paper_id"]: p["metadata"] for p in papers}
    client, emb = get_client(), get_embeddings()
    if rebuild:
        _drop_collection(client, chunk_collection_name(tag))
        _drop_collection(client, config.PAPER_COLLECTION)

    # Chunk collection: tied to one chunking configuration through its config hash.
    chunks_col = _open_collection(client, chunk_collection_name(tag),
                                  {"config_hash": manifest["config_hash"],
                                   "embedding_model": config.EMBEDDING_MODEL})
    _check_fresh(chunks_col, manifest)
    chunks = load_chunks(tag, papers)
    existing = set(chunks_col.get(include=[])["ids"])
    todo = [c for c in chunks if c["chunk_id"] not in existing]
    est_tokens = sum(c["est_tokens"] for c in todo)
    price = config.APPROX_PRICE_PER_MTOK.get(config.EMBEDDING_MODEL, {}).get("input", 0)
    print(f"Chunks: {len(chunks)} in catalog, {len(existing)} already indexed, {len(todo)} to embed "
          f"(~{est_tokens:,} tokens, ~${est_tokens / 1e6 * price:.2f})")
    _embed_into(chunks_col, emb, [(c["chunk_id"], c["text"], _chunk_metadata(c, meta_by_paper[c["paper_id"]]))
                                  for c in todo], "chunks")
    removed = existing - {c["chunk_id"] for c in chunks}  # chunks of papers no longer included
    if removed:
        chunks_col.delete(ids=sorted(removed))

    # Paper collection: re-embed a paper only when its title/authors/abstract text changed.
    papers_col = _open_collection(client, config.PAPER_COLLECTION, {"embedding_model": config.EMBEDDING_MODEL})
    stored = papers_col.get(include=["metadatas"])
    stored_sha = {pid: m.get("text_sha") for pid, m in zip(stored["ids"], stored["metadatas"])}
    paper_items = []
    for paper in papers:
        text = _paper_text(paper)
        md = _paper_metadata(paper, text)
        if stored_sha.get(paper["paper_id"]) != md["text_sha"]:
            paper_items.append((paper["paper_id"], text, md))
    stale = [pid for pid in stored_sha if pid not in meta_by_paper or pid in {i[0] for i in paper_items}]
    if stale:
        papers_col.delete(ids=stale)
    print(f"Papers: {len(papers)} in catalog, {len(paper_items)} to embed")
    _embed_into(papers_col, emb, paper_items, "papers")

    summary = {
        "built_at": _now(),
        "strategy_tag": tag,
        "config_hash": manifest["config_hash"],
        "embedding_model": config.EMBEDDING_MODEL,
        "chunks": chunks_col.count(),
        "papers": papers_col.count(),
        "embedded_this_run": {"chunks": len(todo), "papers": len(paper_items), "est_tokens": est_tokens},
    }
    index_manifest = _read_json(INDEX_MANIFEST, {})
    index_manifest[tag] = summary
    _write_json(INDEX_MANIFEST, index_manifest)
    print(f"\nIndex '{tag}': {summary['chunks']} chunk vectors, {summary['papers']} paper vectors "
          f"in {config.CHROMA_DIR.name}/")
    return summary


# ─── Search ──────────────────────────────────────────────────────────
class HybridRetriever:
    """BM25 + vector search over chunks, fused per chunk_id by weighted reciprocal rank fusion.

    search(..., paper_ids=[...]) restricts both halves to those papers, which is how the agent
    looks for passages inside papers that PaperIndex.find() picked first."""

    def __init__(self, tag: str | None = None):
        self.tag = tag or default_tag()
        manifest = load_manifest(self.tag)
        self._chunks = load_chunks(self.tag)
        self._by_id = {c["chunk_id"]: c for c in self._chunks}
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self._chunks])
        self._col = _open_collection(get_client(), chunk_collection_name(self.tag), {}, create=False)
        _check_fresh(self._col, manifest)
        self._emb = get_embeddings()

    def search(self, query: str, k: int = config.NUM_RETRIEVED, paper_ids: list[str] | None = None,
               page_ranges: dict[str, list[tuple[int, int]]] | None = None) -> list[dict]:
        """page_ranges {volume_id: [(first_page, last_page)]} limits a volume to papers inside it."""
        allowed = set(paper_ids or [])
        ranges = page_ranges or {}

        def in_scope(chunk: dict) -> bool:
            if not allowed and not ranges:
                return True
            return chunk["paper_id"] in allowed or any(
                first <= chunk["page_start"] <= last for first, last in ranges.get(chunk["paper_id"], []))

        scores = self._bm25.get_scores(tokenize(query))
        candidates = [i for i in range(len(self._chunks)) if scores[i] > 0 and in_scope(self._chunks[i])]
        bm = sorted(candidates, key=lambda i: scores[i], reverse=True)[:config.CANDIDATE_POOL]
        bm_ids = [self._chunks[i]["chunk_id"] for i in bm]

        clauses: list[dict] = [{"paper_id": {"$in": sorted(allowed)}}] if allowed else []
        clauses += [{"$and": [{"paper_id": vid}, {"page_start": {"$gte": first}}, {"page_start": {"$lte": last}}]}
                    for vid, spans in ranges.items() for first, last in spans]
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$or": clauses}
        res = self._col.query(query_embeddings=[self._emb.embed_query(query)],
                              n_results=config.CANDIDATE_POOL, where=where, include=["distances"])
        vec_ids = res["ids"][0]

        # Reciprocal rank fusion, not Lab 7.2's min-max score fusion. Min-max always scored the
        # weakest candidate 0, and raw BM25 and cosine scores live on different scales (cosine
        # similarities cluster around 0.4-0.65, so BM25 dominated). RRF uses only each list's
        # ranks: weight / (RRF_K + rank). Scores are scaled so #1 in both lists is 1.0.
        bm_rank = {cid: r for r, cid in enumerate(bm_ids, 1)}
        vec_rank = {cid: r for r, cid in enumerate(vec_ids, 1)}
        best = (config.WEIGHT_BM25 + config.WEIGHT_VECTOR) / (config.RRF_K + 1)

        def rrf(rank: int | None, weight: float) -> float:
            return weight / (config.RRF_K + rank) if rank else 0.0

        fused = []
        for cid in dict.fromkeys(bm_ids + vec_ids):
            c = self._by_id[cid]
            raw = rrf(bm_rank.get(cid), config.WEIGHT_BM25) + rrf(vec_rank.get(cid), config.WEIGHT_VECTOR)
            fused.append({
                "chunk_id": cid, "paper_id": c["paper_id"], "page_start": c["page_start"],
                "page_end": c["page_end"], "section": c["section"], "text": c["text"],
                "bm25_rank": bm_rank.get(cid), "vector_rank": vec_rank.get(cid),
                "score": round(raw / best, 3),
            })
        fused.sort(key=lambda r: r["score"], reverse=True)
        return fused[:k]


class PaperIndex:
    """Paper-level lookup: catalog filters (author, year, doc type) plus semantic search over
    title + abstract. Author matching is a case-insensitive substring of any author name.

    Papers inside proceedings volumes have no record of their own, so the author and year filters
    also check them (ids like <volume_id>@p826). They have no abstract or vector, so a topic
    query alone cannot match them."""

    def __init__(self):
        self._papers = {p["paper_id"]: p for p in load_papers()}
        self._inner = self._load_inner_papers()
        self._col = _open_collection(get_client(), config.PAPER_COLLECTION, {}, create=False)
        self._emb = None  # created on first semantic query

    def _load_inner_papers(self) -> dict[str, dict]:
        """Catalog-shaped rows for the papers inside volumes; entries with no authors are skipped."""
        rows = {}
        for vid, vol in effective_volumes().items():
            meta = self._papers.get(vid, {}).get("metadata")
            if meta is None:
                continue
            for e in vol["papers"]:
                if e["authors"]:
                    pid = f"{vid}@p{e['start_page']}"
                    rows[pid] = {"paper_id": pid, "volume_id": vid, "file": self._papers[vid]["file"],
                                 "title": e["title"], "authors": e["authors"], "year": meta.get("year"),
                                 "venue": meta.get("title"), "doc_type": "paper_in_volume",
                                 "pages": f"{e['start_page']}-{e['end_page']}", "page_start": e["start_page"],
                                 "page_end": e["end_page"], "abstract": None, "score": None}
        return rows

    def _inner_matches(self, author, year_from, year_to) -> list[dict]:
        rows = []
        for row in self._inner.values():
            year = row["year"]
            if author and not any(author.lower() in a.lower() for a in row["authors"]):
                continue
            if year_from and (year is None or year < year_from):
                continue
            if year_to and (year is None or year > year_to):
                continue
            rows.append(row)
        return sorted(rows, key=lambda r: r["year"] or 0, reverse=True)

    def _matches(self, paper: dict, author, year_from, year_to, doc_type) -> bool:
        meta = paper["metadata"]
        if author:
            names = [*(meta.get("authors") or []), meta.get("first_author") or ""]
            if not any(author.lower() in n.lower() for n in names):
                return False
        year = meta.get("year")
        if year_from and (year is None or year < year_from):
            return False
        if year_to and (year is None or year > year_to):
            return False
        return not doc_type or meta.get("doc_type") == doc_type

    def _row(self, paper_id: str, score: float | None) -> dict:
        p = self._papers[paper_id]
        meta = p["metadata"]
        return {"paper_id": paper_id, "file": p["file"], "title": meta.get("title"),
                "authors": meta.get("authors") or [], "year": meta.get("year"),
                "venue": meta.get("venue"), "doc_type": meta.get("doc_type"), "pages": p["pages"],
                "abstract": meta.get("abstract"), "score": score}

    def get(self, paper_id: str) -> dict | None:
        """Catalog row for one included paper, or None for an unknown id."""
        if paper_id in self._inner:
            return self._inner[paper_id]
        return self._row(paper_id, None) if paper_id in self._papers else None

    def find(self, query: str | None = None, author: str | None = None, year_from: int | None = None,
             year_to: int | None = None, doc_type: str | None = None, k: int = 5) -> list[dict]:
        ids = [pid for pid, p in self._papers.items() if self._matches(p, author, year_from, year_to, doc_type)]
        filters_only = not query and (year_from or year_to)
        inner = self._inner_matches(author, year_from, year_to)[:k] if (author or filters_only) and not doc_type else []
        if not ids:
            return inner
        if not query:  # filters only: newest first
            ids.sort(key=lambda pid: self._papers[pid]["metadata"].get("year") or 0, reverse=True)
            return [self._row(pid, None) for pid in ids[:k]] + inner
        self._emb = self._emb or get_embeddings()
        filtered = len(ids) < len(self._papers)
        res = self._col.query(query_embeddings=[self._emb.embed_query(query)], n_results=min(k, len(ids)),
                              where={"paper_id": {"$in": ids}} if filtered else None, include=["distances"])
        return [self._row(pid, round(1 - d, 3)) for pid, d in zip(res["ids"][0], res["distances"][0])] + inner

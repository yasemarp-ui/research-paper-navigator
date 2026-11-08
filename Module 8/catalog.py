"""Offline catalog step: PDFs -> cleaned page text -> paper metadata -> chunks.

No embeddings are created here. The only API calls are the optional metadata extraction
(one small-model call per paper), and every stage is reused while a PDF's sha256 is unchanged:

  catalog/pages/<paper_id>.json        cleaned text per page (extracted once per PDF)
  catalog/papers.json                  paper-level metadata (independent of chunking)
  catalog/<tag>/chunks/<paper_id>.json chunk records for one chunking strategy
  catalog/<tag>/index.json             ordered chunk ids per paper
  catalog/<tag>/manifest.json          chunking settings, config hash, corpus statistics
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pypdf
from pypdf import PdfReader

import config
from chunking import PAGE_SEPARATOR, build_chunks, get_chunker, is_heading, merge_small_spans, strategy_tag
from prompts import METADATA_SYSTEM
from safety import escape_xml

logging.getLogger("pypdf").setLevel(logging.ERROR)  # silences "Exceeded 5000 form XObject" noise


# ─── Small file helpers ──────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    """Write atomically, so an interrupted run never leaves a half-written catalog file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ─── Text extraction and cleaning ────────────────────────────────────
def clean_page_text(raw: str) -> str:
    """Normalize pypdf output so retrieval and chunking work on sensible text.

    - NFKC turns ligatures into letters ("satisﬁes" -> "satisfies"), which BM25 needs.
    - A word hyphenated across a line break is rejoined ("implemen-\\ntation").
    - pypdf emits no blank lines, so paragraph breaks are inferred: after a heading, before a
      heading, or after a short line that ends a sentence and is followed by a capitalized line.
    """
    text = unicodedata.normalize("NFKC", raw).replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    merged: list[str] = []
    for line in lines:
        if merged and re.search(r"[A-Za-z]-$", merged[-1]) and line[:1].islower():
            merged[-1] = merged[-1][:-1] + line.lstrip()
        else:
            merged.append(line)
    if not merged:
        return ""
    typical = statistics.median([len(ln) for ln in merged if len(ln) > 20] or [80])
    out = [merged[0]]
    for prev, line in zip(merged, merged[1:]):
        ends_sentence = re.search(r"[.!?:][\"')\]]?$", prev) is not None
        short = len(prev) < 0.8 * typical
        starts_new = line[:1].isupper() or line[:1].isdigit() or line[:1] in "[(•"
        if is_heading(prev) or is_heading(line) or (ends_sentence and short and starts_new):
            out.append("\n\n" + line)
        else:
            out.append("\n" + line)
    return "".join(out)


def extract_pages(paper_id: str, pdf_path: Path, sha: str) -> dict:
    """Cleaned page texts + embedded PDF metadata, cached in catalog/pages/<paper_id>.json."""
    cache_path = config.PAGES_DIR / f"{paper_id}.json"
    cached = _read_json(cache_path)
    if cached and cached.get("sha256") == sha and cached.get("clean_version") == config.TEXT_CLEAN_VERSION:
        return cached
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(clean_page_text(page.extract_text() or ""))
        except Exception:  # one unreadable page must not lose the whole paper
            pages.append("")
    meta = reader.metadata or {}
    record = {
        "paper_id": paper_id,
        "sha256": sha,
        "clean_version": config.TEXT_CLEAN_VERSION,
        "pdf_meta": {"title": str(meta.get("/Title") or "").strip(),
                     "author": str(meta.get("/Author") or "").strip()},
        "pages": pages,
    }
    _write_json(cache_path, record)
    return record


def page_offsets(pages: list[str]) -> list[int]:
    """Character offset where each page starts once pages are joined with PAGE_SEPARATOR."""
    offsets, pos = [], 0
    for text in pages:
        offsets.append(pos)
        pos += len(text) + len(PAGE_SEPARATOR)
    return offsets


# ─── Metadata from the filename (DBLP keys) ──────────────────────────
def _two_digit_year(yy: str) -> int:
    return 2000 + int(yy) if int(yy) <= 30 else 1900 + int(yy)


def parse_filename(paper_id: str) -> dict:
    """Venue, first-author surname and year encoded in the filename, when present.

    DBLP keys look like DBLP_conf_cav_JeonQSF15 (venue 'cav', first author 'Jeon', 2015);
    DBLP_conf_cav_2011 is proceedings front matter; 'corr' is arXiv (a preprint).
    """
    out = {"venue_key": None, "first_author": None, "year": None, "doc_type": "paper"}
    m = re.match(r"^(?:DBLP_)?(conf|journals|series)_([A-Za-z0-9]+)_(.+)$", paper_id)
    key = paper_id
    if m:
        out["venue_key"], key = m.group(2), m.group(3)
        if out["venue_key"] == "corr":
            out["doc_type"] = "preprint"
        arxiv = re.fullmatch(r"abs-(\d{2})(\d{2})-\d+", key)  # arXiv id YYMM-NNNNN: no author in the key
        if arxiv:
            out["year"] = 2000 + int(arxiv.group(1))
            return out
        if re.fullmatch(r"(19|20)\d{2}", key):
            out.update(year=int(key), doc_type="front_matter")
            return out
        author = re.match(r"([A-Z][a-z]+(?:-[A-Z][a-z]+)?)", key)
        out["first_author"] = author.group(1) if author else None
        year = re.search(r"(\d{2})[a-z]?$", key)
        if year and not key.isdigit():
            out["year"] = _two_digit_year(year.group(1))
        if not out["first_author"]:  # e.g. 'gupta2025sparling' inside a non-DBLP conf_ key
            loose = re.match(r"([a-z]+)((?:19|20)\d{2})", key)
            if loose:
                out.update(first_author=loose.group(1).capitalize(), year=int(loose.group(2)))
        return out
    loose = re.match(r"([A-Za-z][a-z]+)((?:19|20)\d{2}|\d{2})", key)  # e.g. Ellis22Linguistics, Gupta2024
    if loose:
        y = loose.group(2)
        out.update(first_author=loose.group(1).capitalize(),
                   year=int(y) if len(y) == 4 else _two_digit_year(y))
    return out


# ─── Metadata from a small LLM ───────────────────────────────────────
def make_metadata_llm():
    from langchain_openai import ChatOpenAI  # imported lazily: --no-llm runs need no API key
    return ChatOpenAI(model=config.METADATA_MODEL, api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url=config.OPENROUTER_BASE_URL, timeout=90, max_retries=2,
                      reasoning_effort=config.METADATA_REASONING_EFFORT)


def extract_metadata_llm(llm, paper_id: str, first_text: str) -> tuple[dict, dict]:
    """Ask the metadata model for title/authors/year/venue/doc_type/abstract.
    Returns (parsed JSON or {}, token usage)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    doc = f'<document paper_id="{escape_xml(paper_id)}">\n{escape_xml(first_text)}\n</document>'
    response = llm.invoke([SystemMessage(content=METADATA_SYSTEM), HumanMessage(content=doc)])
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


def _clean_str(value, limit: int = 500) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value[:limit] if value else None


def merge_metadata(fn: dict, pdf_meta: dict, llm: dict, first_text: str) -> tuple[dict, dict, list[str]]:
    """Combine the three sources field by field. Returns (metadata, source per field, warnings).

    Precedence: the filename decides year and front matter (DBLP keys are reliable); the LLM
    decides title, authors, venue name and abstract; embedded PDF metadata and the first page
    are fallbacks."""
    meta, src, warnings = {}, {}, []

    title = _clean_str(llm.get("title"), 300)
    pdf_title = _clean_str(pdf_meta.get("title"), 300)
    if title:
        src["title"] = "llm"
    elif pdf_title and pdf_title.lower() not in {"title", "untitled"}:
        title, src["title"] = pdf_title, "pdf_meta"
    else:
        title, src["title"] = (first_text.split("\n", 1)[0].strip()[:300] or None), "first_line"
        warnings.append("title guessed from first line")
    meta["title"] = title

    # Strip affiliation markers the model copies along with names ("Alex Gu1", "Wen-Ding Li⋆2").
    authors = [re.sub(r"[\s\d⋆∗*†‡§¶,()]+$", "", a) for a in (_clean_str(x, 120) for x in llm.get("authors") or []) if a]
    authors = [a for a in authors if a]
    if authors:
        src["authors"] = "llm"
    elif fn["first_author"]:
        authors, src["authors"] = [fn["first_author"]], "filename"
    meta["authors"] = authors
    meta["first_author"] = fn["first_author"] or (authors[0].split()[-1] if authors else None)

    llm_year = llm.get("year") if isinstance(llm.get("year"), int) and 1950 <= llm.get("year") <= 2035 else None
    if fn["year"]:
        meta["year"], src["year"] = fn["year"], "filename"
        if llm_year and abs(llm_year - fn["year"]) > 1:
            warnings.append(f"year mismatch: filename={fn['year']} llm={llm_year}")
    else:
        meta["year"], src["year"] = llm_year, ("llm" if llm_year else None)

    meta["venue"] = _clean_str(llm.get("venue"), 200)
    meta["venue_key"] = fn["venue_key"]
    src["venue"] = "llm" if meta["venue"] else None

    abstract = _clean_str(llm.get("abstract"), 2000)
    if abstract:
        src["abstract"] = "llm"
    else:
        m = re.search(r"Abstract[.:—\-]?\s*(.{100,2000}?)(?:\n\n|Keywords|1\s+Introduction)", first_text, re.S)
        abstract = " ".join(m.group(1).split()) if m else None
        src["abstract"] = "regex" if abstract else None
    meta["abstract"] = abstract

    llm_type = llm.get("doc_type") if llm.get("doc_type") in {"paper", "preprint", "front_matter", "thesis", "other"} else None
    if fn["doc_type"] == "front_matter":
        meta["doc_type"], src["doc_type"] = "front_matter", "filename"
    elif llm_type and llm_type != "paper":
        meta["doc_type"], src["doc_type"] = llm_type, "llm"
    else:
        meta["doc_type"], src["doc_type"] = fn["doc_type"], "filename"
    return meta, src, warnings


# ─── Stage 1+2: pages and paper metadata ─────────────────────────────
def load_subset(path: Path | None) -> set[str] | None:
    """Paper ids listed one per line (with or without .pdf); '#' starts a comment."""
    if not path:
        return None
    if not path.exists():
        raise SystemExit(f"Subset file not found: {path}")
    ids = {ln.split("#", 1)[0].strip().removesuffix(".pdf") for ln in path.read_text().splitlines()}
    return {i for i in ids if i}


def build_papers_catalog(papers_dir: Path, use_llm: bool = True, subset: set[str] | None = None,
                         limit: int | None = None) -> dict:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {papers_dir}")
    if limit:
        pdfs = pdfs[:limit]
    previous = {p["paper_id"]: p for p in (_read_json(config.PAPERS_FILE, {}) or {}).get("papers", [])}
    llm = make_metadata_llm() if use_llm else None
    usage = {"input": 0, "output": 0, "calls": 0}

    # Pass 1 (local): page text, filename metadata, and which papers still need the model.
    items = []
    for pdf in pdfs:
        paper_id = pdf.stem
        sha = sha256_file(pdf)
        rec = extract_pages(paper_id, pdf, sha)
        prev = previous.get(paper_id)
        # The raw model answer is stored, so changing the merge rules never needs new API calls.
        cached = bool(prev and prev.get("sha256") == sha and "llm_raw" in prev)
        items.append({"pdf": pdf, "paper_id": paper_id, "sha": sha, "rec": rec, "prev": prev, "cached": cached,
                      "first_text": PAGE_SEPARATOR.join(rec["pages"])[: config.METADATA_MAX_CHARS]})

    # Pass 2 (API): metadata calls in parallel, only for papers without a cached result.
    def call_model(item: dict) -> tuple[dict, dict, str]:
        try:
            meta, u = extract_metadata_llm(llm, item["paper_id"], item["first_text"])
            return meta, u, ("llm" if meta else "llm-unparsed")
        except Exception as e:  # one failed call falls back to filename/PDF metadata
            return {}, {"input": 0, "output": 0}, f"llm-error ({type(e).__name__})"

    llm_results = {}
    todo = [it for it in items if llm and not it["cached"]]
    if todo:
        print(f"Metadata model: {len(todo)} call(s), {config.METADATA_WORKERS} in parallel...")
        with ThreadPoolExecutor(max_workers=config.METADATA_WORKERS) as pool:
            for item, (meta, u, how) in zip(todo, pool.map(call_model, todo)):
                llm_results[item["paper_id"]] = (meta, how)
                usage["input"] += u["input"]
                usage["output"] += u["output"]
                usage["calls"] += 1

    # Pass 3 (local): merge sources and assemble the catalog entries.
    papers = []
    for i, item in enumerate(items, 1):
        paper_id, rec, prev, sha, pdf = item["paper_id"], item["rec"], item["prev"], item["sha"], item["pdf"]
        pages, first_text = rec["pages"], item["first_text"]
        if item["cached"]:
            llm_meta, how = prev["llm_raw"], "cached"
        else:
            llm_meta, how = llm_results.get(paper_id, ({}, "skipped"))
        meta, src, warnings = merge_metadata(parse_filename(paper_id), rec["pdf_meta"], llm_meta, first_text)
        if how.startswith("llm-"):
            warnings.append(f"metadata model: {how}")

        chars = sum(len(p) for p in pages)
        if chars < 500:
            warnings = warnings + ["very little extractable text (scanned PDF?)"]
        papers.append({
            "paper_id": paper_id,
            "file": pdf.name,
            "sha256": sha,
            "pages": len(pages),
            "chars": chars,
            "page_offsets": page_offsets(pages),
            "include": subset is None or paper_id in subset,
            "metadata": meta,
            "metadata_source": src,
            "warnings": sorted(set(warnings)),
            **({"llm_raw": llm_meta} if how in ("llm", "cached") else {}),
        })
        print(f"[{i:>3}/{len(pdfs)}] {paper_id:<45} pages={len(pages):<4} metadata={how}")

    if subset:
        missing = subset - {p["paper_id"] for p in papers}
        if missing:
            print(f"[warning] {len(missing)} subset id(s) not found among the PDFs: {sorted(missing)[:5]}")

    price = config.APPROX_PRICE_PER_MTOK.get(config.METADATA_MODEL, {"input": 0, "output": 0})
    catalog = {
        "catalog_version": config.CATALOG_VERSION,
        "created_at": _now(),
        "papers_dir": papers_dir.name,  # name only: no local absolute paths in a committed file
        "pypdf_version": pypdf.__version__,
        "text_clean_version": config.TEXT_CLEAN_VERSION,
        "metadata_model": config.METADATA_MODEL,
        "metadata_usage_this_run": {
            **usage,
            "approx_cost_usd": round(usage["input"] / 1e6 * price["input"] + usage["output"] / 1e6 * price["output"], 4),
        },
        "papers": papers,
    }
    from provenance import apply_to_catalog  # cached corrections and overrides; no API calls
    apply_to_catalog(catalog)
    _write_json(config.PAPERS_FILE, catalog)
    print(f"\nWrote {config.PAPERS_FILE.relative_to(config.BASE_DIR)}: {len(papers)} papers, "
          f"{sum(p['include'] for p in papers)} included; metadata calls this run: {usage['calls']}")
    return catalog


# ─── Stage 3: chunks for one strategy ────────────────────────────────
def config_hash(chunking: dict) -> str:
    """Changes whenever anything that affects chunk text or ids changes."""
    basis = {"chunking": chunking, "clean": config.TEXT_CLEAN_VERSION, "catalog": config.CATALOG_VERSION,
             "chunker": config.CHUNKER_VERSION, "min_chunk_chars": config.MIN_CHUNK_CHARS}
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:12]


def _spread(values: list[float]) -> dict:
    if not values:
        return {}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def build_chunk_catalog(strategy: str, chunk_size: int, overlap: int) -> dict:
    if overlap >= chunk_size:
        raise SystemExit(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    catalog = _read_json(config.PAPERS_FILE)
    if not catalog:
        raise SystemExit("No catalog/papers.json yet — run the 'catalog' command first.")
    chunker = get_chunker(strategy)
    tag = strategy_tag(strategy, chunk_size, overlap)
    out_dir = config.CATALOG_DIR / tag
    chunks_dir = out_dir / "chunks"
    if chunks_dir.exists():  # chunking is deterministic and cheap, so rebuild from scratch
        shutil.rmtree(chunks_dir)

    index, all_chunks, included_chunks = {}, [], 0
    for paper in catalog["papers"]:
        rec = _read_json(config.PAGES_DIR / f"{paper['paper_id']}.json")
        if not rec or rec.get("sha256") != paper["sha256"]:
            raise SystemExit(f"Page text for {paper['paper_id']} is missing or stale — rerun 'catalog'.")
        text = PAGE_SEPARATOR.join(rec["pages"])
        spans = merge_small_spans(chunker(text, paper["page_offsets"], chunk_size, overlap),
                                  config.MIN_CHUNK_CHARS)
        chunks = build_chunks(paper["paper_id"], text, paper["page_offsets"], spans, tag)
        _write_json(chunks_dir / f"{paper['paper_id']}.json",
                    {"paper_id": paper["paper_id"], "strategy": tag, "chunks": chunks})
        index[paper["paper_id"]] = [c["chunk_id"] for c in chunks]
        all_chunks.extend(chunks)
        if paper["include"]:
            included_chunks += len(chunks)

    chunking = {"strategy": strategy, "chunk_size": chunk_size, "overlap": overlap}
    manifest = {
        "catalog_version": config.CATALOG_VERSION,
        "strategy_tag": tag,
        "chunking": chunking,
        "config_hash": config_hash(chunking),
        "created_at": _now(),
        "pypdf_version": catalog["pypdf_version"],
        "text_clean_version": config.TEXT_CLEAN_VERSION,
        "chunker_version": config.CHUNKER_VERSION,
        "min_chunk_chars": config.MIN_CHUNK_CHARS,
        "stats": {
            "papers": len(catalog["papers"]),
            "included_papers": sum(p["include"] for p in catalog["papers"]),
            "pages": sum(p["pages"] for p in catalog["papers"]),
            "chunks": len(all_chunks),
            "included_chunks": included_chunks,
            "est_tokens": sum(c["est_tokens"] for c in all_chunks),
            "chunk_chars": _spread([c["n_chars"] for c in all_chunks]),
            "pages_per_chunk": _spread([c["page_end"] - c["page_start"] + 1 for c in all_chunks]),
            "paragraphs_per_chunk": _spread([c["n_paragraphs"] for c in all_chunks]),
            "chunks_with_section": sum(1 for c in all_chunks if c["section"]),
            "doc_types": dict(Counter(p["metadata"]["doc_type"] for p in catalog["papers"])),
        },
    }
    _write_json(out_dir / "index.json", {"strategy_tag": tag, "papers": index})
    _write_json(out_dir / "manifest.json", manifest)
    s = manifest["stats"]
    print(f"\nChunking '{tag}': {s['chunks']} chunks from {s['papers']} papers "
          f"({s['included_chunks']} in included papers), ~{s['est_tokens']:,} tokens; "
          f"median chunk {s['chunk_chars'].get('median')} chars")
    return manifest

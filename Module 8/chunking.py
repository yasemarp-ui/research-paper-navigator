"""Pluggable chunking strategies.

A chunker only decides WHERE to cut: it receives a paper's cleaned text (pages joined by a
blank line) plus the character offset where each page starts, and returns character spans
(start, end). build_chunks() turns those spans into chunk records with the same metadata for
every strategy, so adding a strategy never touches the catalog format, indexing, or the agent.

To add a strategy: write a function with the ChunkerFn signature and register it in CHUNKERS.
"""
from __future__ import annotations

import bisect
import re
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

Span = tuple[int, int]
ChunkerFn = Callable[[str, list[int], int, int], list[Span]]

PAGE_SEPARATOR = "\n\n"  # how catalog.py joins pages into one paper text

# A heading is a short line such as "3.2 Sketch Search", "Abstract" or "RELATED WORK".
# Commas, e-mail signs and a trailing period rule out affiliations and ordinary sentences.
_NAMED_HEADING = re.compile(
    r"^(abstract|introduction|background|related work|evaluation|experiments?|discussion|"
    r"conclusions?|future work|references|bibliography|acknowledge?ments?|appendix\b.*)$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+){0,3}\.?|[A-H](\.\d+){1,3})\s+[A-Z][A-Za-z][^\n]*$")
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 \-:]{3,60}$")


def is_heading(line: str) -> bool:
    line = line.strip()
    if not 3 <= len(line) <= 80 or line.endswith(".") or "," in line or "@" in line:
        return False
    return bool(_NAMED_HEADING.match(line) or _NUMBERED_HEADING.match(line) or _CAPS_HEADING.match(line))


def strategy_tag(strategy: str, chunk_size: int, overlap: int) -> str:
    """Folder and collection name for one chunking configuration, e.g. 'recursive-3000-500'."""
    return f"{strategy}-{chunk_size}-{overlap}"


# ─── Span helpers ────────────────────────────────────────────────────
def _trim(text: str, span: Span) -> Span | None:
    """Shrink a span to exclude surrounding whitespace; None if nothing is left."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _windows(start: int, end: int, chunk_size: int, overlap: int) -> list[Span]:
    """Fixed-size windows over [start, end) that overlap by `overlap` characters."""
    step = max(1, chunk_size - overlap)
    spans, pos = [], start
    while pos < end:
        stop = min(end, pos + chunk_size)
        spans.append((pos, stop))
        if stop == end:
            break
        pos += step
    return spans


def _paragraph_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    end = len(text) if end is None else end
    spans = []
    for m in re.finditer(r"\S(?:.|\n(?!\n))*", text[start:end]):
        spans.append((start + m.start(), start + m.end()))
    return spans


def _finish(text: str, spans: list[Span]) -> list[Span]:
    return [s for s in (_trim(text, sp) for sp in spans) if s]


def _enforce_overlap(text: str, spans: list[Span], overlap: int) -> list[Span]:
    """Make each chunk start `overlap` characters before the previous one ends.

    Boundary-aware splitters only carry over whole pieces that fit inside the overlap, so with
    long paragraphs the real overlap is often zero. This extends each start back to give the
    requested overlap, moved forward to the next word boundary so no chunk begins mid-word.
    A chunk can therefore be up to chunk_size + overlap characters long."""
    if overlap <= 0 or not spans:
        return spans
    out = [spans[0]]
    for start, end in spans[1:]:
        prev_end = out[-1][1]
        target = max(out[-1][0] + 1, prev_end - overlap)
        if start > target:
            while target < start and not text[target - 1].isspace():
                target += 1
            start = target
        out.append((start, end))
    return out


# ─── Strategies ──────────────────────────────────────────────────────
def fixed_chunker(text: str, page_offsets: list[int], chunk_size: int, overlap: int) -> list[Span]:
    """Plain fixed-size windows that ignore structure — the simplest baseline to compare against."""
    return _finish(text, _windows(0, len(text), chunk_size, overlap))


def recursive_chunker(text: str, page_offsets: list[int], chunk_size: int, overlap: int) -> list[Span]:
    """Whole-paper split that prefers paragraph, then line, then sentence, then word boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    spans = []
    for doc in splitter.create_documents([text]):
        start = doc.metadata.get("start_index", -1)
        if start < 0:  # the splitter could not locate the piece; skip rather than guess an offset
            continue
        spans.append((start, start + len(doc.page_content)))
    return _finish(text, _enforce_overlap(text, _finish(text, spans), overlap))


def page_chunker(text: str, page_offsets: list[int], chunk_size: int, overlap: int) -> list[Span]:
    """One chunk per page (exact page citations); pages longer than chunk_size are windowed."""
    spans = []
    for i, start in enumerate(page_offsets):
        end = page_offsets[i + 1] - len(PAGE_SEPARATOR) if i + 1 < len(page_offsets) else len(text)
        if end - start <= chunk_size:
            spans.append((start, end))
        else:
            spans.extend(_windows(start, end, chunk_size, overlap))
    return _finish(text, spans)


def paragraph_chunker(text: str, page_offsets: list[int], chunk_size: int, overlap: int) -> list[Span]:
    """Pack whole paragraphs up to chunk_size, never cutting mid-paragraph unless one paragraph is
    itself too long. Overlap carries the trailing paragraphs (up to `overlap` chars) forward."""
    spans: list[Span] = []
    current: list[Span] = []

    def emit() -> None:
        if current:
            spans.append((current[0][0], current[-1][1]))

    for para in _paragraph_spans(text):
        p_len = para[1] - para[0]
        if p_len > chunk_size:  # an oversized paragraph is windowed on its own
            emit()
            current = []
            spans.extend(_windows(para[0], para[1], chunk_size, overlap))
            continue
        if current and para[1] - current[0][0] > chunk_size:
            emit()
            carried: list[Span] = []
            for prev in reversed(current):
                if current[-1][1] - prev[0] > overlap:
                    break
                carried.insert(0, prev)
            current = carried
        current.append(para)
    emit()
    return _finish(text, _enforce_overlap(text, _finish(text, spans), overlap))


CHUNKERS: dict[str, ChunkerFn] = {
    "recursive": recursive_chunker,
    "page": page_chunker,
    "paragraph": paragraph_chunker,
    "fixed": fixed_chunker,
}


def merge_small_spans(spans: list[Span], min_chars: int) -> list[Span]:
    """Fold chunks shorter than min_chars (stray page headers, captions) into the previous chunk,
    or into the next one when the paper starts with a tiny chunk. Applied after any strategy, so
    a merged chunk can be somewhat longer than chunk_size."""
    if min_chars <= 0:
        return spans
    out: list[Span] = []
    for start, end in spans:
        if out and end - start < min_chars:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        elif out and out[-1][1] - out[-1][0] < min_chars:  # tiny first chunk: absorb into this one
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def get_chunker(name: str) -> ChunkerFn:
    if name not in CHUNKERS:
        raise SystemExit(f"Unknown chunking strategy {name!r}. Choose one of: {', '.join(CHUNKERS)}")
    return CHUNKERS[name]


# ─── Shared chunk metadata ───────────────────────────────────────────
def find_headings(text: str) -> list[tuple[int, str]]:
    """(offset, heading) for every heading-like line, in document order."""
    headings, pos = [], 0
    for line in text.split("\n"):
        if is_heading(line):
            headings.append((pos, line.strip()))
        pos += len(line) + 1
    return headings


def build_chunks(paper_id: str, text: str, page_offsets: list[int], spans: list[Span],
                 tag: str) -> list[dict]:
    """Turn spans into chunk records. Pages are 1-based; `section` is the last heading that
    starts at or before the chunk (a heuristic guess, empty when none is found)."""
    headings = find_headings(text)
    heading_offsets = [h[0] for h in headings]
    chunks = []
    for seq, (start, end) in enumerate(spans):
        body = text[start:end]
        h = bisect.bisect_right(heading_offsets, start) - 1
        chunks.append({
            "chunk_id": f"{paper_id}#c{seq:04d}",
            "paper_id": paper_id,
            "strategy": tag,
            "seq": seq,
            "page_start": bisect.bisect_right(page_offsets, start),
            "page_end": bisect.bisect_right(page_offsets, end - 1),
            "char_start": start,
            "char_end": end,
            "n_chars": len(body),
            "n_lines": body.count("\n") + 1,
            "n_paragraphs": len([p for p in body.split("\n\n") if p.strip()]),
            "est_tokens": round(len(body) / 4),
            "section": headings[h][1] if h >= 0 else "",
            "text": body,
        })
    return chunks

"""Research Paper Navigator — command-line entry point.

  python paper_navigator.py catalog [PAPERS_DIR] [--subset FILE] [--no-llm] [--limit N]
                                    [--chunker NAME] [--chunk-size N] [--overlap N]
  python paper_navigator.py chunk   [--chunker NAME] [--chunk-size N] [--overlap N]
  python paper_navigator.py provenance [--no-llm]
  python paper_navigator.py index   [--chunker NAME] [--chunk-size N] [--overlap N] [--rebuild]
  python paper_navigator.py search  QUERY [--paper ID ...] [--k N]
  python paper_navigator.py search  [QUERY] --papers [--author NAME] [--year-from Y] [--year-to Y]
  python paper_navigator.py chat    [--no-cache] [--no-router] [--quiet]
  python paper_navigator.py ask     QUESTION [--no-cache] [--no-router] [--quiet]
  python paper_navigator.py eval    [--limit N] [--skip-safety]

'catalog' extracts page text, builds paper metadata and chunks for one strategy. 'chunk'
re-chunks from the saved page text only — no PDF parsing, no API calls — so trying another
chunking strategy takes seconds. 'index' embeds whatever is not stored yet (rerun it to resume).
'search' runs retrieval on its own, without the agent, to check what the index returns.
'chat' talks to the agent; 'ask' answers one question (no clarification prompts) and prints
the path it took and the token cost. 'eval' compares baseline and optimized runs and writes
the evidence log; --limit N is a cheap trial with a _trial suffix on its files.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

import config
from chunking import CHUNKERS, strategy_tag


def _add_chunking_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chunker", choices=sorted(CHUNKERS), default=config.CHUNKING["strategy"])
    p.add_argument("--chunk-size", type=int, default=config.CHUNKING["chunk_size"])
    p.add_argument("--overlap", type=int, default=config.CHUNKING["overlap"])


def _require_key(hint: str = "") -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"OPENROUTER_API_KEY is not set (see .env.example){hint}.")


def _search(args: argparse.Namespace) -> None:
    from retrieval import HybridRetriever, PaperIndex
    if args.papers:
        rows = PaperIndex().find(args.query, author=args.author, year_from=args.year_from,
                                 year_to=args.year_to, k=args.k)
        for r in rows:
            authors = ", ".join(r["authors"][:3]) + (" et al." if len(r["authors"]) > 3 else "")
            score = "" if r["score"] is None else f"{r['score']:.3f}  "
            print(f"{score}{r['year']}  [{r['doc_type']}] {r['title']}\n        {authors} · {r['file']}")
        print(f"\n{len(rows)} paper(s)")
        return
    if not args.query:
        raise SystemExit("search needs a QUERY (or use --papers with filters).")
    retriever = HybridRetriever(strategy_tag(args.chunker, args.chunk_size, args.overlap))
    for r in retriever.search(args.query, k=args.k, paper_ids=args.paper):
        pages = f"p.{r['page_start']}" + (f"-{r['page_end']}" if r["page_end"] != r["page_start"] else "")
        snippet = " ".join(r["text"].split())[:220]
        ranks = f"bm25 #{r['bm25_rank'] or '-'} · vec #{r['vector_rank'] or '-'}"
        print(f"{r['score']:.3f} ({ranks})  {r['chunk_id']}  {pages}"
              f"  §{r['section'] or '-'}\n        {snippet}...")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Research Paper Navigator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("catalog", help="extract text + metadata, then chunk")
    p_cat.add_argument("papers_dir", nargs="?", type=Path, default=config.PAPERS_DIR_DEFAULT)
    p_cat.add_argument("--subset", type=Path, default=None,
                       help="file of paper ids to mark include=true (default: all papers)")
    p_cat.add_argument("--no-llm", action="store_true", help="skip the metadata model (no API calls)")
    p_cat.add_argument("--limit", type=int, default=None, help="only the first N PDFs (testing)")
    _add_chunking_args(p_cat)

    p_chunk = sub.add_parser("chunk", help="re-chunk from saved page text (no PDFs, no API)")
    _add_chunking_args(p_chunk)

    p_prov = sub.add_parser("provenance", help="check metadata, find papers inside volumes, write the review file")
    p_prov.add_argument("--no-llm", action="store_true", help="reapply cached results and overrides only")

    p_index = sub.add_parser("index", help="embed chunks and papers into Chroma (resumable)")
    _add_chunking_args(p_index)
    p_index.add_argument("--rebuild", action="store_true", help="drop the collections and embed everything again")

    p_search = sub.add_parser("search", help="try retrieval without the agent")
    p_search.add_argument("query", nargs="?", default=None)
    _add_chunking_args(p_search)
    p_search.add_argument("--paper", action="append", default=None, help="limit to this paper id (repeatable)")
    p_search.add_argument("--k", type=int, default=config.NUM_RETRIEVED)
    p_search.add_argument("--papers", action="store_true", help="search papers (title + abstract) instead of chunks")
    p_search.add_argument("--author", default=None)
    p_search.add_argument("--year-from", type=int, default=None)
    p_search.add_argument("--year-to", type=int, default=None)

    p_chat = sub.add_parser("chat", help="talk to the agent")
    p_ask = sub.add_parser("ask", help="answer one question and show the path and token cost")
    p_ask.add_argument("question")
    for p in (p_chat, p_ask):
        _add_chunking_args(p)
        p.add_argument("--no-cache", action="store_true", help="skip the semantic cache")
        p.add_argument("--no-router", action="store_true", help="always run the full agent")
        p.add_argument("--quiet", action="store_true", help="hide the agent's step-by-step log")

    p_eval = sub.add_parser("eval", help="baseline vs optimized, retrieval, citations, safety")
    _add_chunking_args(p_eval)
    p_eval.add_argument("--limit", type=int, default=None, help="only the first N questions (trial run)")
    p_eval.add_argument("--skip-safety", action="store_true", help="skip the injection checks")

    args = parser.parse_args()

    if args.command == "catalog":
        from catalog import build_chunk_catalog, build_papers_catalog, load_subset
        if not args.no_llm:
            _require_key(" or rerun with --no-llm")
        build_papers_catalog(args.papers_dir, use_llm=not args.no_llm,
                             subset=load_subset(args.subset), limit=args.limit)
        build_chunk_catalog(args.chunker, args.chunk_size, args.overlap)
    elif args.command == "chunk":
        from catalog import build_chunk_catalog
        build_chunk_catalog(args.chunker, args.chunk_size, args.overlap)
    elif args.command == "provenance":
        from provenance import build_provenance
        if not args.no_llm:
            _require_key(" or rerun with --no-llm")
        build_provenance(use_llm=not args.no_llm)
    elif args.command == "index":
        from retrieval import build_index
        _require_key()
        build_index(strategy_tag(args.chunker, args.chunk_size, args.overlap), rebuild=args.rebuild)
    elif args.command == "search":
        _require_key()
        _search(args)
    elif args.command in ("chat", "ask"):
        _require_key()
        from agent import build_agent
        agent = build_agent(strategy_tag(args.chunker, args.chunk_size, args.overlap), debug=not args.quiet,
                            use_cache=not args.no_cache, use_router=not args.no_router)
        if args.command == "chat":
            agent.chat()
        else:
            print(f"\n{agent.query(args.question)}\n")
            print(f"Path: {agent.last_trace.get('path')}")
            c = agent.last_trace.get("citations") or {}
            if c:
                print(f"Citation check: {len(c['cited'])} source(s) cited, "
                      f"{len(c['invalid'])} invalid id(s) dropped{' — NO CITATIONS' if c['uncited'] else ''}")
        print("\nToken usage this session:")
        agent.print_token_usage()
    elif args.command == "eval":
        _require_key()
        from evaluate import run
        run(strategy_tag(args.chunker, args.chunk_size, args.overlap), args.limit, args.skip_safety)


if __name__ == "__main__":
    main()

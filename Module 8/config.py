"""Central configuration for the Research Paper Navigator.

Everything a reviewer might want to change lives here: paths, models, and the default
chunking strategy. Command-line flags in paper_navigator.py override the chunking values.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ─── Corpus ──────────────────────────────────────────────────────────
# Default location of the course dataset in this project; pass a path on the command line
# to use a different folder.
PAPERS_DIR_DEFAULT = BASE_DIR.parent / "Module 1" / "Capstone_Checkpoint_1_1_starter" / "ResearchPapers"
SUBSET_FILE_DEFAULT = BASE_DIR / "subset.txt"

# ─── Catalog layout ──────────────────────────────────────────────────
CATALOG_DIR = BASE_DIR / "catalog"
PAGES_DIR = CATALOG_DIR / "pages"          # cleaned page text per paper (gitignored: full paper text)
PAPERS_FILE = CATALOG_DIR / "papers.json"  # paper-level metadata, independent of chunking
CATALOG_VERSION = 1
TEXT_CLEAN_VERSION = 1                     # bump when clean_page_text() changes, to re-extract
CHUNKER_VERSION = 2                        # bump when a chunker's behavior changes (part of the config hash)

# ─── Chunking (default strategy; see chunking.CHUNKERS for the options) ──
# overlap is exact: each chunk repeats the last `overlap` characters of the previous one
# (moved to a word boundary), so a chunk can be up to chunk_size + overlap characters.
CHUNKING = {"strategy": "recursive", "chunk_size": 3000, "overlap": 100}
MIN_CHUNK_CHARS = 300                      # shorter chunks are merged into a neighbor (0 disables)

# ─── Models (via OpenRouter) ─────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
METADATA_MODEL = "openai/gpt-5-nano"       # one call per paper, reads the first pages only
METADATA_MAX_CHARS = 6000                  # roughly pages 1-2
METADATA_REASONING_EFFORT = "low"          # nano is a reasoning model; low effort is plenty for copying fields
METADATA_WORKERS = 6                       # parallel metadata calls

# ─── Provenance (correct citations) ──────────────────────────────────
PROVENANCE_VERSION = 1                     # bump when the provenance prompts change, to re-check
PROVENANCE_MODEL = "openai/gpt-5.4-mini"   # second, stronger extraction of every paper's metadata
PROVENANCE_REASONING_EFFORT = "low"
PROVENANCE_MAX_CHARS = 4000                # first pages read for the metadata check
WRONG_FILE_YEAR_GAP = 5                    # document year this far from the file name's: a different work
VOLUME_MIN_PAGES = 100                     # longer documents are searched for papers inside them
VOLUME_PAGE_CHARS = 3500                   # text of one candidate first page sent to the model
METADATA_CHECK_FILE = CATALOG_DIR / "metadata_check.json"
VOLUME_PAPERS_FILE = CATALOG_DIR / "volume_papers.json"
OVERRIDES_FILE = CATALOG_DIR / "metadata_overrides.json"   # manual corrections, always win
REVIEW_FILE = CATALOG_DIR / "metadata_review.csv"

# ─── Index and retrieval ─────────────────────────────────────────────
CHROMA_DIR = BASE_DIR / "chroma_papers"    # vector store (gitignored; rebuilt with the 'index' command)
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBED_BATCH = 100                          # chunks per embedding request
PAPER_COLLECTION = "papers"                # one vector per paper: title + authors + abstract
PAPER_SUMMARY_MAX_CHARS = 1500
NUM_RETRIEVED = 5                          # chunks returned per search
CANDIDATE_POOL = 30                        # per half before fusion (Lab 7.2 used 10 for 153 emails)
WEIGHT_BM25 = 0.5                          # weights of the two rankings in reciprocal rank fusion
WEIGHT_VECTOR = 0.5
RRF_K = 60                                 # standard RRF constant; larger flattens the rank differences

# ─── Agent (adapted from the Lab 7.2 solution) ───────────────────────
LLM_MODEL = "openai/gpt-5.4-mini"          # plan, answer and the direct path
ROUTER_MODEL = "openai/gpt-5-nano"         # router, cache validation, satisfaction check
ROUTER_REASONING_EFFORT = "low"            # short JSON verdicts need little reasoning
FALLBACK_MODEL = "openai/gpt-5.4"          # full agent when a direct answer is unsatisfactory
MAX_ITERATIONS = 5
FIND_PAPER_K = 15                          # papers returned per catalog lookup
PLAN_PREVIEW_CHARS = 600                   # the planner sees previews, not whole chunks
ANSWER_MAX_CHUNKS = 12                     # best-scoring chunks passed to the answer step
CACHE_DIR = BASE_DIR / "semantic_cache_papers"   # gitignored; delete it to measure a cold cache
CACHE_MAX_SIZE = 400
CACHE_SIMILARITY_THRESHOLD = 0.85          # high on purpose: a wrong cached answer is worse than a miss

# ─── Approximate prices (USD per 1M tokens) — rough estimates only ───
# Verify live prices at https://openrouter.ai/models before quoting real numbers.
APPROX_PRICE_PER_MTOK = {
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},  # checked on OpenRouter 2026-11-01
    "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai/gpt-5.4-mini": {"input": 0.20, "output": 0.80},
    "openai/gpt-5.4": {"input": 2.50, "output": 10.0},
}

# Research Paper Navigator

A retrieval-augmented question-answering agent over a collection of 153 research PDFs on
program synthesis and related topics. It answers questions about what the papers say, finds
papers by author, year or title, and cites every claim with the file and page it came from.

Built for the capstone of the MIT xPRO course *Context and RAG Engineering* (Assignment 8.1).
The system lives in [`Module 8/`](Module%208/), the course's module for the final project.

## How it works

**Offline (run once):**

1. `pypdf` extracts and cleans the text of every page.
2. A small model (`gpt-5-nano`) reads the first pages of each PDF and extracts title, authors,
   year, venue and abstract. A second model (`gpt-5.4-mini`) checks that metadata, flags files
   whose content doesn't match their name, and finds the 162 papers inside 5 proceedings volumes.
   Manual corrections in `catalog/metadata_overrides.json` always win.
3. The text is split into chunks of about 3,000 characters with an exact 100-character overlap
   (7,642 chunks).
4. Chunks and one summary per paper are embedded (`text-embedding-3-small`) into Chroma, and a
   BM25 keyword index is built over the same chunks.

**Online (each question):**

1. The question is sanitized against prompt injection.
2. A semantic cache returns an earlier answer to the same question, after a model confirms the
   match.
3. A router (`gpt-5-nano`) sends simple questions to a direct path (one hybrid search and a
   `gpt-5.4-mini` answer, with a quality check that escalates weak answers to `gpt-5.4`) and
   everything else to a LangGraph agent that plans, searches chunks, looks papers up in the
   catalog, or asks for clarification.
4. Hybrid search merges the BM25 and vector rankings with reciprocal rank fusion.
5. The model cites sources by id only; code turns those ids into references from the catalog,
   so a reference can't be invented. Retrieved text is passed inside XML tags and treated as data.

## Repository layout

```
requirements.txt          pinned dependencies
.env.example              template for the API key
Module 8/
  paper_navigator.py      command-line entry point (all commands below)
  config.py               paths, models, chunking, retrieval and price settings
  catalog.py              page text extraction and paper metadata
  provenance.py           metadata check, papers inside volumes, overrides
  chunking.py             pluggable chunkers
  retrieval.py            hybrid search and paper lookup
  agent.py                cache, router, direct path and LangGraph agent
  prompts.py              all system prompts
  safety.py               input sanitizing and XML escaping
  evaluate.py             evaluation runs and safety checks
  eval_questions.json     the 24 evaluation questions
  catalog/                paper metadata, corrections and the chunk index (see below)
  checkpoint_8_1_evaluation.log, eval_*.csv   evaluation evidence
```

Generated data is not committed: page text (`catalog/pages/`), chunk text
(`catalog/*/chunks/`), the vector store (`chroma_papers/`) and the cache
(`semantic_cache_papers/`). The setup steps below rebuild them.

## Setup

Requirements: Python 3.12 and an [OpenRouter](https://openrouter.ai) API key. Tested on macOS
(Apple Silicon).

```bash
git clone https://github.com/yasemarp-ui/research-paper-navigator.git
cd research-paper-navigator
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your OPENROUTER_API_KEY
```

The only environment variable is `OPENROUTER_API_KEY`. Never commit `.env`; it is in
`.gitignore`.

### The dataset

The PDFs are course material and are not in this repository. Put the 153 `ResearchPapers` PDFs
from the course here:

```
Module 1/Capstone_Checkpoint_1_1_starter/ResearchPapers/
```

or pass another folder to the `catalog` command. On macOS, if a downloaded folder gives
`getcwd: Operation not permitted`, clear the quarantine flag:
`xattr -dr com.apple.quarantine <folder>`.

### Build the index

Run from `Module 8/`:

```bash
cd "Module 8"
python paper_navigator.py catalog --no-llm   # page text and chunks; reuses the committed metadata, no API calls
python paper_navigator.py index              # embeds chunks and papers (about $0.10, a few minutes; rerun to resume)
```

`catalog` without `--no-llm` also calls the metadata model for any paper whose metadata is
missing. The committed `catalog/papers.json`, `metadata_check.json` and `volume_papers.json`
hold the model results, so the full metadata pipeline doesn't need to run again.

## Usage

```bash
python paper_navigator.py chat                                   # interactive
python paper_navigator.py ask "What is program sketching?"       # one question, shows path and cost
python paper_navigator.py search "counterexample-guided synthesis"          # retrieval only
python paper_navigator.py search --papers --author "Solar-Lezama" --year-from 2020
python paper_navigator.py eval                                   # full evaluation (about $0.20)
python paper_navigator.py eval --limit 3                         # cheap trial run
```

`chat` and `ask` take `--no-cache` and `--no-router` to turn the optimizations off, and
`--quiet` to hide the step-by-step log. Settings such as models, chunk size and the number of
retrieved chunks are in `config.py`.

## Evaluation

`evaluate.py` runs the 24 questions in `eval_questions.json` three times: a baseline (no cache,
no router, always the full agent) and two optimized runs (cache and router on, each with an empty
cache). Results from the final run (2026-11-04):

| | Baseline | Optimized 1 | Optimized 2 |
|---|---|---|---|
| Answered, errors | 24/24, 0 | 24/24, 0 | 24/24, 0 |
| Approx. cost | $0.0628 | $0.0547 (−13%) | $0.0570 (−9%) |
| Time | 135 s | 176 s | 190 s |
| Expected paper retrieved / cited | 95% / 91% | 95% / 80% | 90% / 80% |
| Out-of-scope questions declined | 100% | 100% | 100% |
| Invalid citation ids | 0 | 0 | 0 |

- **Graded answers:** 21 pass, 2 partial, 1 fail, graded by hand in `eval_grading_yp.csv`.
- **Citations:** 15 of 15 spot-checked references were correct (`eval_citation_check.csv`).
- **Safety:** an injection probe was sanitized and the real question was answered; a passage
  with planted instructions was followed in 0 of 3 runs.

Costs are estimates from token counts and the prices in `config.py`. Answers vary from run to
run.

## Known limitations

- **Volumes are single documents.** Five proceedings volumes contain 162 papers. The papers can
  be found by author and year, but they have no abstract or vector of their own, so an answer
  may cite the volume rather than the paper inside it.
- **Author lists are capped.** A catalog lookup returns at most 15 records plus 15 papers from
  volumes, without saying the list is partial.
- **The corpus has errors that are kept on purpose.** 18 files don't contain what their names
  say (one is the AES standard), some works appear under two file names, and some files are
  front matter or rebuttals. Metadata follows what the PDF prints, typos included; years use the
  arXiv stamp when the PDF has one, so they can differ from the venue year.
- **Routing isn't perfect.** The router can send a comparison to the single-search path, which
  gives a one-sided answer.
- **Out-of-scope questions can be the expensive case.** On the direct path the quality check
  rejects the "not in the papers" answer and escalates to `gpt-5.4`, which limits the savings.
- **Section names and paragraph counts are heuristics** from PDF text.
- **Evaluation is small:** 24 questions, one run per configuration.

# EvidencePath AI

**Find the passage. Check the source. Understand the answer.**

**A local document research assistant for AI reasoning and problem solving.**

EvidencePath AI retrieves passages from eight public research papers and uses a local language model to answer questions using that evidence. It displays source filenames, PDF page numbers, and similarity scores so answers can be checked.

## What it covers

Transformers, RAG, BERT, sentence embeddings, vector search, Chain-of-Thought, Tree of Thoughts, and ReAct. This application explains research using retrieved documents; it does not implement the papers' agent or reasoning algorithms.

## Quick start

Use Python 3.12. From this project folder:

```bash
python -m pip install -r requirements.txt
python scripts/download_documents.py
python evidencepath.py
```

Enter `STOP` to finish. First use downloads the embedding and generation models from Hugging Face. Inference runs locally on CPU; no paid API key is required.

## Repeat the benchmark

```bash
python -u evidencepath.py < benchmark_questions.txt > Console_Output.txt 2>&1
```

Windows PowerShell:

```powershell
Get-Content benchmark_questions.txt | python -u evidencepath.py 2>&1 | Tee-Object Console_Output.txt
```

The benchmark has five answerable questions across five papers and one unrelated question. Model answers must be compared with their evidence. See [evaluation](docs/EVALUATION.md) for the recorded behavior and limitations.

## How it works

PDF/TXT loading → character chunks → normalized embeddings → FAISS search → displayed evidence → grounded prompt → local FLAN-T5 answer.

- Chunk size: 450 characters; overlap: 75.
- Embeddings: all-MiniLM-L6-v2, 384 dimensions.
- Search: exact inner-product FAISS search over normalized vectors; top two passages.
- Generator: google/flan-t5-base, local CPU inference, deterministic four-beam search.
- Low similarity: an experimental 0.4 warning; generation still runs.

## Project files

| File | Purpose |
|---|---|
| `evidencepath.py` | Commented RAG application |
| `scripts/download_documents.py` | Downloads original PDFs and checks their hashes |
| `SOURCES.md` | Original paper URLs and provenance |
| `benchmark_questions.txt` | Repeatable benchmark input |
| `docs/EVALUATION.md` | Observations and limitations |
| `PROJECT.md` | Motivation and document rationale |

## Limits

Similarity is relevance, not confidence or truth. The model may give incomplete, verbose, or unsupported answers. Prompts longer than 512 tokens are truncated, potentially dropping later evidence. The benchmark is small and does not establish production reliability. Scientific tables and PDF extraction can affect retrieval.

Source PDFs are fetched from arXiv and retain their original publication rights. They are not redistributed in this repository folder. The project contains public research provenance and reproducible benchmark inputs.

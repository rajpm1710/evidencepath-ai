# EvidencePath AI

## Project brief

EvidencePath AI is a local research assistant that turns a collection of public AI papers into a searchable, evidence-grounded question-answering tool. It helps readers investigate how language models represent text, retrieve knowledge, and approach reasoning and problem solving.

## The problem

Relevant facts are scattered across lengthy publications. A fluent model answer alone does not reveal whether the source supports it. EvidencePath retrieves passages first, exposes their filenames and page numbers, and supplies those passages to a local generator.

## What the project demonstrates

- Document ingestion and traceable PDF page metadata.
- Character-based chunking and overlap.
- Normalized 384-dimensional sentence embeddings.
- FAISS vector search with displayed similarity scores.
- Grounded prompting and local FLAN-T5 generation.
- Repeatable testing with answerable and unanswerable questions.

## Research collection

Eight public papers connect Transformer foundations, BERT, Sentence-BERT, RAG, and vector search with Chain-of-Thought, Tree of Thoughts, and ReAct. The assistant retrieves explanations about these methods; it does not execute their agent frameworks.

## Technical stack

Python, PyTorch, Hugging Face Transformers, SentenceTransformers, FAISS, pypdf, and LangChain text splitters. Inference runs locally on CPU without a paid API key.

## Evaluation snapshot

The evaluated collection contains 163 PDF pages and 1,722 chunks. The selected benchmark includes five answerable questions across five papers and one unrelated question. Core answers are supported by retrieved evidence, although some responses are verbose or incomplete. The unrelated question receives an insufficient-information response. An exploratory ReAct question produced an incorrect answer, illustrating the limits of a small local generator.

A similarity threshold prints a warning rather than forcing a refusal. The generator uses deterministic four-beam search. Context is limited to 512 tokens, so longer evidence can be truncated. These results do not establish production reliability.

## Explore the project

- [Setup and usage](README.md)
- [Public research sources](SOURCES.md)
- [Evaluation and limitations](docs/EVALUATION.md)
- [Project architecture and document rationale](docs/PROJECT_OVERVIEW.md)

Repository name: `evidencepath-ai`.

# AI Research Assistant for Reasoning and Problem Solving

## Problem and purpose

Research on AI problem solving is spread across long papers. Readers need a way to find relevant passages, identify the original source, and judge whether an explanation is supported. This project applies a retrieval-augmented generation workflow to eight public papers covering language models, evidence retrieval, and reasoning methods.

The application supports document-based research questions about these methods. It does not implement the Chain-of-Thought, Tree of Thoughts, or ReAct algorithms as autonomous agents.

## Why these eight documents belong together

| Document | Role in the collection |
|---|---|
| Attention Is All You Need | Transformer and attention foundations |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Combining document retrieval with generation |
| BERT | Language representations and question-answering evaluation |
| Sentence-BERT | Sentence embeddings and semantic retrieval |
| Billion-scale similarity search with GPUs | Vector similarity search foundations |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Intermediate reasoning steps for complex tasks |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | Exploring and evaluating alternative paths during problem solving |
| ReAct: Synergizing Reasoning and Acting in Language Models | Combining reasoning with actions and external information |

Source URLs, original filenames, page counts, and hashes are provided in `../SOURCES.md`.

## Implementation

The application loads eight public PDFs and keeps their source and page metadata. Basic normalization repairs PDF line wraps and ligatures. The recursive splitter uses 450-character chunks and 75-character overlap. The application creates normalized 384-dimensional embeddings and indexes them in FAISS. For each question, it displays two retrieved passages, their page numbers, and their similarity scores. A grounded prompt supplies those passages to a local FLAN-T5-base model. The session continues until STOP.

Retrieval uses exact FAISS similarity search. Source routing, filtering, and custom reranking are potential future extensions.

## Evaluation plan

The recorded run includes five answerable questions and one deliberately unanswerable question. The questions cover both AI foundations and problem-solving methods. The complete console transcript records the actual evidence and generated responses.

Answers are assessed by comparison with the retrieved evidence, rather than by fluency or similarity score alone. `EVALUATION.md` describes the final observed behavior, and `EVALUATION.md` checks source traceability, prompt length, and output completeness.

## Reliability and limitations

A relevant passage does not guarantee a correct answer. PDF tables, references, and figure text can produce misleading retrieval matches. A small local generator may copy too much text, omit relevant information, or supply an unsupported answer. The unanswerable test is included to expose this behavior. The final unanswerable prompt contains 530 tokens, so the tokenizer truncates it to 512. Its question and instructions remain at the start, but part of the later evidence is omitted. These results support a research prototype, not a claim of production reliability.

Possible future improvements include more varied test questions, better handling of tables, a more capable local generator, and evaluated retrieval and abstention methods. They are not claimed as implemented features.


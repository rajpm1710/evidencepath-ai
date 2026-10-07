# Evaluation

The final application was run on eight original public PDFs: 163 pages, 1,722 chunks, and 384-dimensional embeddings. Five answerable questions and one unanswerable question were processed. Every question displays two passages, source filenames, page numbers, scores, and a generated answer. The program exits normally on STOP.

Direct model loading uses deterministic four-beam search to match the original pipeline's decoding defaults. A score below 0.4 warns about weak evidence; it does not force an answer or bypass generation. The threshold is experimental.

## Test 1

**Question:** How does SQuAD 2.0 extend the SQuAD 1.1 problem definition in the BERT paper?

**Actual answer:** The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic. We use a simple approach to extend the SQuAD v1.1 BERT model for this task [Source 2: 03_bert.pdf, page 6] . We find that BERT LARGE significantly outperforms BERTBASE across all tasks, especially those with very little training data. The effect of model size is explored more thoroughly in Section 5.2. 4.2

## Test 2

**Question:** What does the Transformer use instead of recurrence and convolutions in Attention Is All You Need?

**Actual answer:** self-attention

## Test 3

**Question:** What network structures does Sentence-BERT use to derive semantically meaningful sentence embeddings?

**Actual answer:** siamese / triplet

## Test 4

**Question:** What is a chain of thought in the Chain-of-Thought Prompting paper?

**Actual answer:** resembles a solution and can interpreted as one, but we still opt to call it a chain of thought to better capture the idea that it mimics a step-by-step thought process for arriving at the answer (and also, solutions/explanations typically come after the final answer (Narang et al., 2020; Wiegreffe et al., 2022; Lampinen et al., 2022, inter alia)

## Test 5

**Question:** What can Tree of Thoughts do when necessary to make global choices?

**Actual answer:** explore multiple reasoning paths over thoughts

## Test 6

**Question:** What is the exact surface temperature of the planet Neptune today?

**Actual answer:** The documents do not contain enough information.

## Assessment and limits

The five answerable tests have supported core answers. The BERT and Chain-of-Thought responses are verbose, and the Tree of Thoughts response does not fully describe lookahead and backtracking. The unrelated question receives an appropriate insufficient-information response. This small selected benchmark does not demonstrate general reliability. Earlier exploratory ReAct testing produced an incorrect answer.

Five answerable prompts fit within 512 tokens. The unanswerable prompt is 530 tokens and is truncated to 512, potentially omitting later evidence. Similarity is relevance rather than confidence or truth.

The recorded answers above came from an actual execution. Run the benchmark command in the README to capture a complete transcript for your environment.

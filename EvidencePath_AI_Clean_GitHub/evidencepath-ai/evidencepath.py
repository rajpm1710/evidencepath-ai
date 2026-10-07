"""EvidencePath AI: local, evidence-grounded research assistance.

Load PDF/TXT documents, retrieve relevant passages with FAISS, and answer
questions with a local language model. Run scripts/download_documents.py
before first use. Enter STOP to end the interactive session.
"""


from pathlib import Path
import re
import unicodedata

import torch
# Limit CPU threads so this small classroom application remains practical locally.
torch.set_num_threads(2)

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# SETTINGS
# ============================================================

# Locate documents relative to this Python file.
#
# This allows the program to find the folder even if the terminal
# is currently working from a different directory.

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "documents"


# Use focused passages with overlap to preserve nearby context.

# Sizes are measured in characters, not words or tokens.
# Overlap can preserve context when a passage crosses a chunk boundary.
CHUNK_SIZE = 450
CHUNK_OVERLAP = 75


# Retrieve two passages to keep the evidence context focused.
# A nearest match is not proof that the document contains the answer.
TOP_K = 2

# Experimental warning threshold; it does not decide whether an answer exists.
MIN_RETRIEVAL_SCORE = 0.4


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"  # Larger local model for clearer answers.


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(
    folder: Path
) -> list[dict]:

    documents = []
    # Index only supported files; ignore unrelated project files.
    supported_files = [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}]


    # The program cannot continue until the documents
    # folder has been created.

    if not folder.exists():

        raise FileNotFoundError(
            f"Could not find '{folder}'.\n\n"
            "Create a folder named 'documents' in the "
            "same folder as this Python file. Then place "
            "PDF and/or TXT documents inside that folder."
        )


    for file_path in sorted(
        folder.iterdir()
    ):

        # Filenames and PDF page numbers stay attached to the extracted text.
        # This lets the console show where each retrieved passage came from.
        suffix = file_path.suffix.lower()


        # ----------------------------------------------------
        # PDF FILES
        # ----------------------------------------------------

        if suffix == ".pdf":

            reader = PdfReader(
                file_path
            )


            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""

                # Repair PDF line wrapping and typographic ligatures before
                # chunking. This preserves wording rather than summarizing it.
                text = unicodedata.normalize("NFKC", text)
                text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
                text = re.sub(r"\s+", " ", text).strip()


                if text:

                    documents.append(
                        {
                            "text": text,
                            "source": file_path.name,
                            "page": page_number,
                        }
                    )


        # ----------------------------------------------------
        # TEXT FILES
        # ----------------------------------------------------

        elif suffix == ".txt":

            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()


            if text:

                documents.append(
                    {
                        "text": text,
                        "source": file_path.name,
                        "page": None,
                    }
                )


    # The folder may exist but still contain no usable documents.

    if not documents:

        raise ValueError(
            "No readable PDF or TXT content was found in "
            "documents. Add readable PDF and/or "
            "TXT documents and run the program again."
        )


    return documents


# ============================================================
# SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(
    documents: list[dict]
) -> list[dict]:


    # Create a RecursiveCharacterTextSplitter using CHUNK_SIZE
    # and CHUNK_OVERLAP.

    # Prefer paragraph and sentence boundaries before splitting on spaces
    # or individual characters. This helps keep each chunk understandable.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )


    chunks = []


    for document in documents:

        pieces = splitter.split_text(
            document["text"]
        )


        # Each chunk inherits its source metadata; splitting does not
        # discard the information needed to trace an answer to its evidence.
        for chunk_number, piece in enumerate(
            pieces,
            start=1
        ):

            chunks.append(
                {
                    "text": piece,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk": chunk_number,
                }
            )


    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks: list[dict]
):


    # Load the EMBEDDING_MODEL using SentenceTransformer.

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )


    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # Encode the chunk text.
    #
    # Keep normalize_embeddings=True.

    # Unit-length vectors make inner product equal cosine similarity.
    # FAISS expects the numerical matrix in float32 format.
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


    return embedding_model, embeddings


# ============================================================
# BUILD FAISS VECTOR STORE
# ============================================================

def build_vector_store(
    embeddings: np.ndarray
):


    #
    # 1. Determine the number of values in each embedding.
    # 2. Create a FAISS IndexFlatIP index.
    # 3. Add the embeddings to the index.

    # Rows represent chunks; columns represent embedding dimensions.
    # Read the dimension from the matrix instead of hard-coding it.
    dimension = embeddings.shape[1]


    # IndexFlatIP performs an exact inner-product search over all vectors.
    # Passage text and source metadata remain in the separate chunks list.
    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings
    )


    return index


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve(
    question: str,
    embedding_model,
    index,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:


    # Convert the question into an embedding.

    # Use the same model and normalization for questions and documents
    # so both vectors occupy the same space and can be compared.
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


    # Search the FAISS index for the TOP_K closest chunks.

    # FAISS returns scores and row positions. The positions point back
    # into chunks because vectors were added in that same order.
    # Limit k to the number of indexed vectors.
    scores, positions = index.search(
        question_embedding,
        min(top_k, index.ntotal)
    )


    results = []


    for score, position in zip(
        scores[0],
        positions[0]
    ):

        if position == -1:

            continue


        result = dict(
            chunks[position]
        )


        # Similarity measures relevance; it is not an accuracy percentage.
        result["score"] = float(
            score
        )


        results.append(
            result
        )


    return results


# ============================================================
# DISPLAY RETRIEVED EVIDENCE
# ============================================================

def display_retrieved_passages(
    results: list[dict]
) -> None:


    print(
        "\n" + "=" * 72
    )


    print(
        "RETRIEVED EVIDENCE"
    )


    print(
        "=" * 72
    )


    for number, result in enumerate(
        results,
        start=1
    ):

        page_text = (
            f" | Page {result['page']}"
            if result["page"] is not None
            else " | Page N/A (TXT)"
        )


        print(
            f"\nResult {number}"
            f"\nSource: {result['source']}{page_text}"
            f"\nSimilarity score: {result['score']:.4f}"
            f"\n{'-' * 72}"
            f"\n{result['text']}"
        )


# ============================================================
# BUILD GROUNDED PROMPT
# ============================================================

def build_prompt(
    question: str,
    results: list[dict]
) -> str:


    # Keep source labels with the evidence supplied to the language model.
    # Retrieval adds knowledge to the prompt; it does not retrain the model.
    context_sections = []


    for number, result in enumerate(
        results,
        start=1
    ):

        page_text = (
            f", page {result['page']}"
            if result["page"] is not None
            else ""
        )


        context_sections.append(
            f"[Source {number}: "
            f"{result['source']}"
            f"{page_text}]\n"
            f"{result['text']}"
        )


    context = "\n\n".join(
        context_sections
    )


    # Complete the grounded prompt.
    #
    # Keep the QUESTION and INSTRUCTIONS before the EVIDENCE.
    #
    # The prompt must tell the model:
    #
    #   - answer the question directly
    #   - use only the supplied evidence
    #   - do not use outside knowledge
    #   - do not invent missing information
    #   - state when the documents do not contain enough information

    # Put the question and grounding rules before the evidence.
    # Explicitly request an insufficient-information response for missing facts.
    # These instructions encourage grounding but cannot guarantee it.
    prompt = f"""Question: {question}
Instructions: Answer briefly using only the evidence. Do not use outside knowledge or invent missing facts. Answer the question directly. If the answer is absent, say "The documents do not contain enough information."
Evidence:
{context}
Answer:""".strip()


    return prompt


# ============================================================
# CREATE HUGGING FACE GENERATOR
# ============================================================

def create_generator():


    # Load the Hugging Face model and tokenizer directly.

    # Hugging Face downloads the model on first use and caches it.
    # The model is moved to CPU for local inference without a paid API service.
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    model.to("cpu")
    model.eval()
    generator = (model, tokenizer)


    return generator


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    generator,
    prompt: str
) -> str:


    # Generate the answer.
    #
    # Keep do_sample=False so results are more reproducible.

    # Deterministic four-beam search matches the previous pipeline defaults.
    # do_sample=False avoids random sampling.
    # max_new_tokens limits the answer length, not the evidence length.
    # Truncation protects the 512-token input limit. Long evidence may
    # lose text at the end, which is why the question and rules come first.
    model, tokenizer = generator
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=120, do_sample=False, num_beams=4)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():


    print(
        "\nEvidencePath AI"
    )


    print(
        "Evidence-grounded research for AI reasoning and problem solving"
    )


    print(
        "-" * 72
    )


    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Generation model: {GENERATION_MODEL}")
    print(f"Chunk size: {CHUNK_SIZE} characters; overlap: {CHUNK_OVERLAP}; top_k: {TOP_K}")

    documents = load_documents(
        DATA_DIR
    )


    print(
        f"\nLoaded {len({d['source'] for d in documents})} documents; "
        f"{len(documents)} document pages/text records."
    )


    chunks = split_documents(
        documents
    )


    print(
        f"Created {len(chunks)} chunks."
    )


    embedding_model, embeddings = create_embeddings(
        chunks
    )


    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )


    index = build_vector_store(
        embeddings
    )


    print(
        f"FAISS contains "
        f"{index.ntotal} vectors."
    )


    print(
        "\nLoading local language model..."
    )


    generator = create_generator()


    print(
        "Model loaded."
    )


    print(
        "\nAsk questions about your documents."
    )


    print(
        "Enter STOP when you are finished.\n"
    )


    # Build the index and load the models once, then reuse them for questions.
    # STOP ends the interactive session without rebuilding anything.
    while True:


        question = input(
            "Question: "
        ).strip()


        print(question, flush=True)  # Preserve typed questions in redirected output.

        # Enter STOP to end the program.

        if question.upper() == "STOP":

            print(
                "Program ended."
            )

            break


        if not question:

            continue


        results = retrieve(
            question=question,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
        )


        # Display evidence before generation so a reader can independently
        # judge retrieval quality and compare it with the generated answer.
        display_retrieved_passages(
            results
        )


        best_score = results[0]["score"] if results else None
        if best_score is None or best_score < MIN_RETRIEVAL_SCORE:
            print("\nWarning: Retrieved evidence has low similarity; the model will still evaluate it.")

        prompt = build_prompt(
            question,
            results
        )


        answer = generate_answer(
            generator,
            prompt
        )


        print(
            "\n" + "=" * 72
        )


        print(
            "GENERATED ANSWER"
        )


        print(
            "=" * 72
        )


        print(
            answer
        )


        print(
            "\n" + "-" * 72 + "\n"
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
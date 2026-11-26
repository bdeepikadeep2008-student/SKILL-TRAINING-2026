"""
DocuMind AI - PDF Question Answering Chatbot
Python + Ollama + Llama 3.2 + Nomic embeddings + ChromaDB + PyPDF + Gradio

Google Colab preparation (run in notebook cells):
    !apt-get update -qq
    !apt-get install -y -qq zstd
    !pip install -q ollama chromadb pypdf gradio
    !curl -fsSL https://ollama.com/install.sh | sh
    !ollama pull llama3.2:3b
    !ollama pull nomic-embed-text

Start Ollama with `ollama serve`, then run:
    !python documind_ai.py

This app processes one PDF at a time. Its ChromaDB data is in memory and
will be cleared when the Python process / Colab runtime stops.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

import chromadb
import gradio as gr
import ollama
from pypdf import PdfReader

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="documind_documents",
    metadata={"hnsw:space": "cosine"},
)

document_ready = False
document_name = ""
chat_history: list[dict[str, str]] = []


def check_ollama() -> tuple[bool, str]:
    """Check the Ollama server and required model availability."""
    try:
        result = ollama.list()
        models = result.get("models", []) if isinstance(result, dict) else result.models
        names = []
        for model in models:
            if isinstance(model, dict):
                names.append(model.get("name", ""))
            else:
                names.append(getattr(model, "model", "") or getattr(model, "name", ""))

        missing = [
            required for required in (LLM_MODEL, EMBED_MODEL)
            if not any(name == required or name.startswith(required + ":") for name in names)
        ]
        if missing:
            return False, "Missing Ollama models: " + ", ".join(missing)
        return True, "Ollama is connected and both models are available."
    except Exception as exc:
        return False, f"Cannot connect to Ollama. Start `ollama serve`. Details: {exc}"


def make_embedding(text: str) -> list[float]:
    """Convert text into a vector using Nomic embeddings."""
    result = ollama.embed(model=EMBED_MODEL, input=text)
    vectors = result.get("embeddings", []) if isinstance(result, dict) else result.embeddings
    if not vectors:
        raise RuntimeError("The embedding model returned no embedding.")
    return vectors[0]


def extract_pdf_pages(pdf_path: str) -> list[dict[str, Any]]:
    """Extract selectable text from each PDF page."""
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def split_into_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split page text into overlapping chunks while retaining page numbers."""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page["page"]})
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def process_pdf(pdf_file) -> str:
    """Read a PDF, create embeddings, and store chunks in ChromaDB."""
    global document_ready, document_name, chat_history

    if pdf_file is None:
        return "Please upload a PDF file first."

    pdf_path = getattr(pdf_file, "name", pdf_file)
    document_name = os.path.basename(pdf_path)

    connected, status = check_ollama()
    if not connected:
        return status

    try:
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            return "No selectable text found. This may be a scanned PDF and require OCR."

        chunks = split_into_chunks(pages)
        if not chunks:
            return "No text chunks could be created from this PDF."

        old = collection.get()
        if old and old.get("ids"):
            collection.delete(ids=old["ids"])

        ids, documents, metadatas, embeddings = [], [], [], []
        for index, chunk in enumerate(chunks):
            ids.append(f"{uuid.uuid4().hex}_{index}")
            documents.append(chunk["text"])
            metadatas.append({
                "source": document_name,
                "page": int(chunk["page"]),
                "chunk": index,
            })
            embeddings.append(make_embedding(chunk["text"]))

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        document_ready = True
        chat_history = []

        return (
            f"Successfully processed **{document_name}**!\n\n"
            f"- Pages containing text: {len(pages)}\n"
            f"- Text chunks stored: {len(chunks)}\n\n"
            "Open the **Chat with PDF** tab and ask a question."
        )
    except Exception as exc:
        document_ready = False
        return f"Error while processing the PDF: {exc}"


def retrieve_context(question: str) -> list[dict[str, Any]]:
    """Retrieve the most relevant chunks for a question."""
    query_vector = make_embedding(question)
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metadata = result.get("metadatas", [[]])[0]
    return [
        {"text": doc, "metadata": meta or {}}
        for doc, meta in zip(docs, metadata)
    ]


def answer_question(question: str, history=None):
    """Use retrieved PDF context to generate a grounded answer with Llama."""
    global chat_history

    question = (question or "").strip()
    if not question:
        return chat_history, "Please type a question."
    if not document_ready or collection.count() == 0:
        return chat_history, "Please upload and process a PDF first."

    try:
        matches = retrieve_context(question)
        if not matches:
            return chat_history, "I could not find relevant information in the PDF."

        context = "\n\n".join(
            f"[Page {item['metadata'].get('page', 'unknown')}]\n{item['text']}"
            for item in matches
        )
        prompt = f"""You are DocuMind AI, a helpful PDF question-answering assistant.
Answer using only the PDF context below. Be clear and beginner-friendly.
If the answer is not in the context, say you could not find it in the PDF.
Do not invent facts. Mention page numbers when useful.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

        result = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.2},
        )
        answer = (
            result.get("response", "").strip()
            if isinstance(result, dict)
            else result.response.strip()
        )
        if not answer:
            answer = "The language model returned an empty response."

        chat_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return chat_history, ""
    except Exception as exc:
        return chat_history, f"Error while generating the answer: {exc}"


def clear_chat():
    """Clear chat messages without deleting the processed document."""
    global chat_history
    chat_history = []
    return [], ""


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# DocuMind AI\n"
            "### Ask questions about your PDF using Retrieval-Augmented Generation (RAG)"
        )

        with gr.Tab("Upload PDF"):
            gr.Markdown("### Step 1: Upload and process your document")
            pdf_input = gr.File(
                label="Choose a PDF file",
                file_types=[".pdf"],
                type="filepath",
            )
            process_button = gr.Button("Process Document", variant="primary")
            process_status = gr.Markdown()
            process_button.click(process_pdf, inputs=pdf_input, outputs=process_status)

        with gr.Tab("Chat with PDF"):
            gr.Markdown("### Step 2: Ask a question")
            chatbot = gr.Chatbot(label="DocuMind Assistant", height=450)
            question_input = gr.Textbox(
                label="Ask your question",
                placeholder="Example: What is Python?",
                lines=2,
            )
            with gr.Row():
                send_button = gr.Button("Send Message", variant="primary")
                clear_button = gr.Button("Clear Chat")
            answer_status = gr.Markdown()

            send_button.click(
                answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, answer_status],
            ).then(lambda: "", inputs=None, outputs=question_input)

            question_input.submit(
                answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, answer_status],
            ).then(lambda: "", inputs=None, outputs=question_input)

            clear_button.click(
                clear_chat,
                inputs=None,
                outputs=[chatbot, answer_status],
            )

        gr.Markdown(
            "---\n"
            "**Powered by:** Llama 3.2 · Ollama · Nomic Embeddings · "
            "ChromaDB · PyPDF · Gradio\n\n"
            "**Privacy:** Do not upload confidential documents to a publicly shared Gradio link."
        )
    return demo


if __name__ == "__main__":
    connected, message = check_ollama()
    print(message)
    if not connected:
        raise SystemExit("DocuMind AI cannot start until Ollama and its models are ready.")

    app = build_app()
    app.queue().launch(share=True, debug=True, theme=gr.themes.Soft())

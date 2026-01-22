import dspy
import os
from dotenv import load_dotenv

load_dotenv()
import json
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from dspy import ChainOfThought
from pathlib import Path
import random


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # wrapper for sentence-transformers
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


import mlflow

mlflow.dspy.autolog(
    log_compiles=True,  # Track optimization process
    log_evals=True,  # Track evaluation results
    log_traces_from_compile=True,  # Track program traces during optimization
)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("med-ai-workshop")


question = "What is a language model in one sentence?"
local_lm = dspy.LM(
    model="ollama_chat/qwen3:4b", api_base="http://localhost:11434", api_key=""
)
print(local_lm(question))

"""
Build a FAISS vector DB from two local PDF papers using LangChain.
Outputs a directory "faiss_index" with the persisted vectorstore.
"""

DIABETES_PDF_PATHS = [
    "docs/diabets1.pdf",
    "docs/diabets2.pdf",
]  # <-- put your two PDF filenames here
COPD_PDF_PATHS = ["docs/copd1.pdf", "docs/copd2.pdf"]
OUTPUT_DIABETES_FAISS_DIR = "faiss_index/diabetes"
OUTPUT_COPD_FAISS_DIR = "faiss_index/copd"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# chunk settings (tweak for your needs)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200


# - We use `PyPDFLoader` to read each PDF into page-level `Document` objects.
# - We add `source` and `page` metadata to every `Document` to keep track of provenance.
# - Keeping page granularity helps trace answers back to the original PDF.


def load_pdfs(paths):
    """Load PDFs into LangChain Document objects (keeps page-level granularity)."""
    all_docs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        loader = PyPDFLoader(str(p))
        # load returns a list of Document objects (one per page typically)
        pages = loader.load()
        # add a source filename into metadata for traceability
        for i, doc in enumerate(pages):
            # ensure a copy of metadata dict (avoid mutating shared objects)
            meta = dict(doc.metadata or {})
            meta["source"] = str(p.name)
            meta["page"] = i
            doc.metadata = meta
        all_docs.extend(pages)
    return all_docs


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks (keeps metadata)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # split_documents returns list[Document] (with page_content and metadata)
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vector_store(
    chunks, model_name=EMBEDDING_MODEL, save_dir=OUTPUT_COPD_FAISS_DIR
):
    """create embeddings and store them in faiss vector store and persist to disk"""
    hf_emb = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": "cpu"}
    )

    print(
        "Creating FAISS vector store from ",
        len(chunks),
        "chunks. This may take a while...",
    )
    vector_store = FAISS.from_documents(chunks, hf_emb)
    vector_store.save_local(save_dir)
    print(f"Saved FAISS vectorstore to: {save_dir}")
    return vector_store, hf_emb


print("Loading Diabetes PDFs...")
docs = load_pdfs(DIABETES_PDF_PATHS)
print(f"Loaded {len(docs)} page-documents from {len(DIABETES_PDF_PATHS)} PDFs.")

print("Chunking Diabetes documents...")
chunks = chunk_documents(docs)
print(
    f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
)

diabetes_vectorstore, diabetes_embeddings = build_vector_store(
    chunks, save_dir=OUTPUT_DIABETES_FAISS_DIR
)

print("Loading COPD PDFs...")
docs = load_pdfs(COPD_PDF_PATHS)
print(f"Loaded {len(docs)} page-documents from {len(COPD_PDF_PATHS)} PDFs.")

print("Chunking COPD documents...")
chunks = chunk_documents(docs)
print(
    f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
)

copd_vectorstore, copd_embeddings = build_vector_store(
    chunks, save_dir=OUTPUT_COPD_FAISS_DIR
)


def diabetes_vector_search_tool(query: str, k: int = 3) -> str:
    """
    A tool for the ReAct agent.
    Performs vector search and returns a formatted string of results.
    """
    results = diabetes_vectorstore.similarity_search_with_score(query, k=k)
    context = ""
    for i, (doc, score) in enumerate(results):
        doc_content = doc.page_content
        context += f"[PASSAGE {i + 1}, score={score:.4f}]\n{doc_content}\\n\\n"
    return context


def copd_vector_search_tool(query: str, k: int = 3) -> str:
    """
    A tool for the ReAct agent.
    Performs vector search and returns a formatted string of results.
    """
    results = copd_vectorstore.similarity_search_with_score(query, k=k)
    context = ""
    for i, (doc, score) in enumerate(results):
        doc_content = doc.page_content
        context += f"[PASSAGE {i + 1}, score={score:.4f}]\n{doc_content}\\n\\n"
    return context


lm = dspy.LM(
    "openrouter/openai/gpt-oss-20b",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=64000,
)
dspy.settings.configure(lm=lm)


class RAGQA(dspy.Signature):
    """You are a helpful assistant. Answer a question using retrieved passages"""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


rag = ChainOfThought(RAGQA)

question = "What is Gestational Diabetes Mellitus (GDM)?"
retrieved_context = diabetes_vector_search_tool(question, k=3)
rag(context=retrieved_context, question=question)

lm.inspect_history(n=1)


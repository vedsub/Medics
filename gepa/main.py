import os
from pathlib import Path

import dspy
from dspy import ChainOfThought
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # wrapper for sentence-transformers
from langchain_community.vectorstores import FAISS
load_dotenv()

"""
Build a FAISS vector DB from two local PDF papers using LangChain.
Outputs a directory "faiss_index" with the persisted vectorstore.
"""

REPO_ROOT = Path(__file__).resolve().parent

DIABETES_PDF_PATHS = [
    REPO_ROOT / "docs" / "diabets1.pdf",
    REPO_ROOT / "docs" / "diabets2.pdf",
]  # <-- put your two PDF filenames here
COPD_PDF_PATHS = [
    REPO_ROOT / "docs" / "copd1.pdf",
    REPO_ROOT / "docs" / "copd2.pdf",
]
OUTPUT_DIABETES_FAISS_DIR = REPO_ROOT / "faiss_index" / "diabetes"
OUTPUT_COPD_FAISS_DIR = REPO_ROOT / "faiss_index" / "copd"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# chunk settings (tweak for your needs)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200


# - We use `PyPDFLoader` to read each PDF into page-level `Document` objects.
# - We add `source` and `page` metadata to every `Document` to keep track of provenance.
# - Keeping page granularity helps trace answers back to the original PDF.


def load_pdfs(paths: list[Path]):
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


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


def get_or_build_vector_store(
    *,
    pdf_paths: list[Path],
    save_dir: Path,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Load an existing persisted FAISS index if present; otherwise build from PDFs and persist.
    """
    save_dir = Path(save_dir)
    index_file = save_dir / "index.faiss"
    if index_file.exists():
        print(f"Loading existing FAISS index from: {save_dir}")
        return FAISS.load_local(
            str(save_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print(f"Building FAISS index (not found on disk) -> {save_dir}")
    print("Loading PDFs...")
    docs = load_pdfs(pdf_paths)
    print(f"Loaded {len(docs)} page-documents from {len(pdf_paths)} PDFs.")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(
        f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )

    print(f"Creating FAISS vector store from {len(chunks)} chunks. This may take a while...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    save_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_dir))
    print(f"Saved FAISS vectorstore to: {save_dir}")
    return vector_store


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


def configure_dspy_lm() -> dspy.LM:
    """
    Default to local Ollama for easy offline testing.
    To use OpenRouter, set BOTH:
      - USE_OPENROUTER=1
      - OPENROUTER_API_KEY=...
    """
    if os.environ.get("USE_OPENROUTER") == "1" and os.environ.get("OPENROUTER_API_KEY"):
        lm = dspy.LM(
            "openrouter/openai/gpt-oss-20b",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model_type="chat",
            cache=False,
            temperature=0.3,
            max_tokens=4096,
        )
    else:
        lm = dspy.LM(
            model="ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            api_key="",
        )
    dspy.settings.configure(lm=lm)
    return lm


class RAGQA(dspy.Signature):
    """You are a helpful assistant. Answer a question using retrieved passages"""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


rag = ChainOfThought(RAGQA)


def ask_diabetes(question: str, *, k: int = 3) -> str:
    results = diabetes_vectorstore.similarity_search_with_score(question, k=k)
    print("\n=== Retrieved context (diabetes) ===")
    context = ""
    for i, (doc, score) in enumerate(results, start=1):
        source = (doc.metadata or {}).get("source", "unknown")
        page = (doc.metadata or {}).get("page", "unknown")
        print(f"\n[PASSAGE {i}] score={score:.4f} source={source} page={page}\n{doc.page_content}")
        context += f"[PASSAGE {i}, score={score:.4f}]\n{doc.page_content}\n\n"

    pred = rag(context=context, question=question)
    print("\n=== Answer ===")
    print(pred.answer)
    return pred.answer


if __name__ == "__main__":
    # Optional MLflow logging (won't crash your script if MLflow server isn't running)
    try:
        import mlflow

        mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "med-ai-workshop"))
    except Exception as e:
        print(f"(mlflow disabled) {e}")

    embeddings = get_embeddings()
    diabetes_vectorstore = get_or_build_vector_store(
        pdf_paths=DIABETES_PDF_PATHS,
        save_dir=OUTPUT_DIABETES_FAISS_DIR,
        embeddings=embeddings,
    )
    copd_vectorstore = get_or_build_vector_store(
        pdf_paths=COPD_PDF_PATHS,
        save_dir=OUTPUT_COPD_FAISS_DIR,
        embeddings=embeddings,
    )

    lm = configure_dspy_lm()

    q = "What is diabetes mellitus?"
    ask_diabetes(q, k=3)


# Load the dataset
with open("docs/qa_pairs_diabets.json", "r") as f:
    qa_diabetes_data = json.load(f)

# Convert to dspy.Example objects
dataset_diabetes = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_diabetes_data]

# shuffle the dataset
random.shuffle(dataset_diabetes)

# Split the dataset as requested
train_size = 20
trainset_diabetes = dataset_diabetes[:train_size]
devset_diabetes = dataset_diabetes[train_size:]

print(f"Loaded {len(dataset_diabetes)} examples.")
print(f"Train set size: {len(trainset_diabetes)}")
print(f"Dev set size: {len(devset_diabetes)}")


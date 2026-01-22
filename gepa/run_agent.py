"""
Run DiabetesAgent without rebuilding FAISS indices.
Uses pre-built vector stores from faiss_index/ directory.
"""
import dspy
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIABETES_FAISS_DIR = SCRIPT_DIR / "faiss_index/diabetes"

# Load pre-built FAISS index (no need to rebuild)
print("Loading pre-built FAISS index...")
hf_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
diabetes_vectorstore = FAISS.load_local(
    str(OUTPUT_DIABETES_FAISS_DIR), hf_emb, allow_dangerous_deserialization=True
)
print("FAISS index loaded!")


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


class ReActSignature(dspy.Signature):
    """You are a helpful assistant. Answer user's question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class DiabetesAgent(dspy.Module):
    def __init__(self):
        # init LLM - using local Ollama model
        self.lm = dspy.LM(
            model="ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            api_key="",
            temperature=0.3,
            cache=False
        )
        dspy.configure(lm=self.lm)
        self.diabetes_agent = dspy.ReAct(ReActSignature, tools=[diabetes_vector_search_tool])

    def forward(self, question: str):
        return self.diabetes_agent(question=question)


if __name__ == "__main__":
    # Create the agent
    diabetes_agent = DiabetesAgent()
    
    # Test with a question
    question = "What are the main treatments for Type 2 diabetes?"
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    result = diabetes_agent(question=question)
    print(f"\nAnswer: {result.answer}")
    
    # Optionally inspect the history
    # diabetes_agent.lm.inspect_history(n=2)

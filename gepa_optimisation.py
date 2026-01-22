"""
GEPA Optimization for DiabetesAgent
Run this after main.py has been executed (or import from main.py)
"""
import dspy
import json
import logging
import os
import random
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
GEPA_DIR = SCRIPT_DIR / "gepa"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIABETES_FAISS_DIR = GEPA_DIR / "faiss_index/diabetes"

# Load pre-built FAISS index
print("Loading pre-built FAISS index...")
hf_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
diabetes_vectorstore = FAISS.load_local(
    str(OUTPUT_DIABETES_FAISS_DIR), hf_emb, allow_dangerous_deserialization=True
)
print("FAISS index loaded!")

# Setup LM
lm = dspy.LM(
    model="ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="",
    temperature=0.3,
    cache=False
)
dspy.settings.configure(lm=lm)

# Load dataset
with open(GEPA_DIR / "docs/qa_pairs_diabets.json", "r") as f:
    qa_diabetes_data = json.load(f)

dataset_diabetes = [
    dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question")
    for item in qa_diabetes_data
]
random.shuffle(dataset_diabetes)

train_size = 20
trainset_diabetes = dataset_diabetes[:train_size]
devset_diabetes = dataset_diabetes[train_size:]

print(f"Loaded {len(dataset_diabetes)} examples.")
print(f"Train set size: {len(trainset_diabetes)}")
print(f"Dev set size: {len(devset_diabetes)}")


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
        self.lm = lm
        self.diabetes_agent = dspy.ReAct(ReActSignature, tools=[diabetes_vector_search_tool])

    def forward(self, question: str):
        return self.diabetes_agent(question=question)


# ===================== GEPA JUDGES =====================

class JudgeConsistency(dspy.Signature):
    """Judge whether the predicted answer matches the gold answer.

    # Instructions:
    - The score should be between 0.0 and 1.0 and based on the similarity of the predicted answer and the gold answer.
    - The justification should be a brief explanation of the score.
    - If the answer doesn't address the question properly, the score should be less than 0.5.
    - If the answer is completely correct, the score should be 1.0. Otherwise, the score should be less than 1.0.
    - Be very strict in your judgement as this is a medical question.
    """
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    justification: str = dspy.OutputField()


class JudgeReactStep(dspy.Signature):
    """Judge whether the next tool call (name + args) is appropriate, well-formed, and relevant.

    - Output a strict score in [0, 1].
    - Provide a brief justification and a yes/no style verdict in justification text.
    """
    question: str = dspy.InputField()
    tool_name: str = dspy.InputField()
    tool_args_json: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    verdict: str = dspy.OutputField()
    justification: str = dspy.OutputField()


def llm_metric_prediction(*args, **kwargs):
    """Metric returning ScoreWithFeedback for GEPA and Evaluate."""
    pred_name = kwargs.get("pred_name")
    pred_trace = kwargs.get("pred_trace")
    if pred_name is not None:
        logger.info(f"metric called for predictor = {pred_name}")

    example = kwargs.get("example") or kwargs.get("gold")
    pred = kwargs.get("pred") or kwargs.get("prediction")
    if example is None and len(args) > 0:
        example = args[0]
    if pred is None and len(args) > 1:
        pred = args[1]

    if pred_name and (pred_name == "react" or pred_name.endswith(".react")) and pred_trace:
        try:
            _, step_inputs, step_outputs = pred_trace[0]
        except Exception:
            step_inputs, step_outputs = {}, {}

        question_text = getattr(example, "question", None) or step_inputs.get("question", "") or ""

        def _get(o, key, default=""):
            if isinstance(o, dict):
                return o.get(key, default)
            return getattr(o, key, default)

        tool_name = _get(step_outputs, "next_tool_name", "")
        tool_args = _get(step_outputs, "next_tool_args", {})

        args_is_dict = isinstance(tool_args, dict)
        has_query = args_is_dict and isinstance(tool_args.get("query"), str) and tool_args.get("query", "").strip() != ""
        k_val = tool_args.get("k") if args_is_dict else None
        k_ok = isinstance(k_val, int) and 1 <= k_val <= 10 or k_val is None
        used_tool = tool_name not in ("", "finish")
        early_finish = tool_name == "finish"

        logger.debug(
            "react-step details | used_tool=%s tool=%s args_keys=%s has_query=%s k=%s early_finish=%s pred_trace_len=%s",
            used_tool, str(tool_name),
            list(tool_args.keys()) if isinstance(tool_args, dict) else type(tool_args).__name__,
            has_query, k_val, early_finish,
            len(pred_trace) if pred_trace else 0,
        )

        heuristics_score = 0.0
        if used_tool:
            heuristics_score += 0.4
        if has_query:
            heuristics_score += 0.4
        if k_ok:
            heuristics_score += 0.1
        if not early_finish:
            heuristics_score += 0.1
        heuristics_score = max(0.0, min(1.0, heuristics_score))

        # LLM judge for the loop step
        tool_args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
        with dspy.settings.context(lm=lm):
            react_judge = dspy.Predict(JudgeReactStep)
            judged = react_judge(
                question=question_text,
                tool_name=str(tool_name),
                tool_args_json=tool_args_json,
            )

        llm_score = getattr(judged, "score", 0.0) or 0.0
        llm_score = max(0.0, min(1.0, llm_score))
        llm_just = getattr(judged, "justification", "") or ""

        total = 0.5 * heuristics_score + 0.5 * llm_score

        logger.info("react-step scores | heuristics=%.3f llm=%.3f total=%.3f", heuristics_score, llm_score, total)

        suggestions = []
        if not used_tool:
            suggestions.append("Select a retrieval tool before finishing.")
        if early_finish:
            suggestions.append("Avoid selecting 'finish' until you have evidence from the retrieval tool.")
        if not args_is_dict:
            suggestions.append("Emit next_tool_args as a valid JSON object.")
        else:
            if not has_query:
                suggestions.append("Include a non-empty 'query' string in next_tool_args.")
            if k_val is not None and (not isinstance(k_val, int) or k_val < 1 or k_val > 10):
                suggestions.append("Choose a reasonable k (e.g., 3–5).")
        if not suggestions:
            suggestions.append("Good step. Keep queries concise and set k=5 by default.")

        feedback_text = (
            f"ReAct step — LLM score: {llm_score:.2f}, heuristics: {heuristics_score:.2f}. "
            + " ".join(suggestions)
            + (f" LLM justification: {llm_just}" if llm_just else "")
        ).strip()

        return ScoreWithFeedback(score=total, feedback=feedback_text)

    # Program-level: judge final answer quality
    if example is None or pred is None:
        return ScoreWithFeedback(score=0.0, feedback="Missing example or pred")

    predicted_answer = getattr(pred, "answer", None) or ""
    if not predicted_answer.strip():
        return ScoreWithFeedback(score=0.0, feedback="Empty prediction")

    with dspy.settings.context(lm=lm):
        judge = dspy.Predict(JudgeConsistency)
        judged = judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=predicted_answer,
        )

    score = getattr(judged, "score", None) or 0.0
    score = max(0.0, min(1.0, score))
    justification = getattr(judged, "justification", "") or ""
    logger.info("answer-level score=%.3f for question='%s'", score, (example.question[:80] + "...") if len(example.question) > 80 else example.question)
    feedback_text = f"Score: {score}. {justification}".strip()
    return ScoreWithFeedback(score=score, feedback=feedback_text)


if __name__ == "__main__":
    # Create the agent
    diabetes_agent = DiabetesAgent()

    # Run evaluation
    evaluator_diabetes = Evaluate(
        devset=devset_diabetes,
        num_threads=4,  # Reduced for local model
        display_progress=True,
        display_table=5,
        provide_traceback=True
    )
    print("Evaluating the baseline ReAct agent...")
    diabetes_baseline_eval = evaluator_diabetes(diabetes_agent, metric=llm_metric_prediction)
    print(f"\nBaseline evaluation result: {diabetes_baseline_eval}")


teacher_lm = dspy.LM(
    model="ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="",
    model_type="chat",
    cache=False,
    temperature=0.3
)




teleprompter = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=2,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
)

optimized_diabetes_agent = teleprompter.compile(student=diabetes_agent, trainset=trainset_diabetes, valset=devset_diabetes)
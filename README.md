# Medics: Multi-Agent Medical Operations System

Medics is a sophisticated multi-agent system designed to answer complex medical questions, specifically focusing on **Chronic Obstructive Pulmonary Disease (COPD)** and **Diabetes**. It leverages **DSPy** for agent orchestration and **GEPA (Gradient Emended Proposition-based Adjustment)** for optimizing agent instructions.

## System Architecture

The system consists of three main components:
1.  **Lead ReAct Agent:** The central orchestrator that decomposes user queries and delegates tasks to specialized sub-agents.
2.  **Diabetes Agent:** A specialized agent equipped with vector search tools for diabetes-related medical literature.
3.  **COPD Agent:** A specialized agent equipped with vector search tools for COPD-related medical literature.

All agents utilize **RAG (Retrieval-Augmented Generation)** techniques, fetching information from domain-specific PDF documents stored in a FAISS vector database.

## Performance & Evaluation

The system has been rigorously evaluated using a custom evaluation framework. The GEPA optimization process yielded a significant **~4% improvement** across key metrics.

### Evaluation Results

| Agent Component | Evaluation Score | Best Validation Score | Best Candidate Index |
| :--- | :--- | :--- | :--- |
| **Diabetes Agent** | **90.72%** | 0.947 | 1 |
| **COPD Agent** | **89.44%** | 0.942 | 2 |
| **Baseline Lead ReAct** | **88.79%** | 0.924 | 1 |

### Optimized Agent Instructions

Through the optimization process, the agents evolved their system instructions to provide more structured and clinically accurate responses. Below is the optimized instruction set for the **Lead ReAct Agent** (Best Candidate):

<details>
<summary><strong>Click to view Optimized Lead ReAct Instructions</strong></summary>

```markdown
## Instruction for the Assistant

You are a **helpful agent** that answers medical questions about COPD and diabetes.  
Your job is to produce a *structured response* that interleaves the following
elements in the exact order:

1. `next_thought` – a short paragraph describing your current reasoning and
   the next step you plan to take.
2. `next_tool_name` – the name of the tool you will call next.  
   Allowed tool names are:
   * `ask_diabetes`
   * `ask_copd`
   * `finish`
3. `next_tool_args` – a JSON object containing the arguments for the chosen
   tool.  The JSON must be syntactically valid.  
   * For `ask_diabetes` and `ask_copd` the argument is a single field
     `question` of type string.
   * For `finish` the argument object is empty `{}`.

After you call a tool, you will receive an `observation` that is appended to
your trajectory.  Use that observation to update your next thought and decide
whether another tool call is needed or if you should finish.

**Important constraints**

- **Do not output anything else** besides the three fields above.  
  No additional text, comments, or tool calls are allowed.
- **Do not use any other tools** (e.g., `browser`, `python`).  
  Only the three tools listed above are permitted.
- The `next_tool_args` must be valid JSON; do not embed it in quotes or
  add stray characters.
- When you are ready to produce the final answer, call the `finish` tool
  with an empty argument object `{}`.

**Domain knowledge you should incorporate**

1. **COPD inpatient protocol for acute exacerbations**  
   * Steroid course: 5 days of systemic corticosteroids (oral prednisone 40–60 mg/d or IV methylprednisolone 40–80 mg/d).  
   * Oxygen target: SpO₂ 88–92 % (or 90–94 % for chronic hypercapnia).  
   * Bronchodilators: Nebulized SABA + ipratropium; add LAMA/LABA if not on maintenance.  
   * Antibiotics: Empiric if sputum purulence or fever; common regimens include amoxicillin‑clavulanate or doxycycline.  
   * Monitoring: Vitals & SpO₂ q4 h, daily ABG if PaCO₂ > 45 mmHg, labs (CBC, electrolytes, CRP).  
   * Discharge planning: Stable vitals, SpO₂ ≥ 88 % on ≤ 4 L/min O₂, no IV antibiotics, provide written action plan, inhaler technique review, smoking cessation counseling, referral to pulmonary rehab.

2. **Diabetes management in steroid‑treated patients**  
   * Immediate hyperglycemia management: Basal‑bolus insulin regimen (e.g., glargine + rapid‑acting insulin at meals).  
   * Monitoring: SMBG 4–6× daily, including a dedicated post‑steroid check 2–3 h after dose.  
   * Adjustment: Increase basal dose or add a short‑acting insulin before steroid dose if post‑steroid glucose >200 mg/dL.  
   * Discharge follow‑up: Medication reconciliation, 2–4 week outpatient visit, HbA1c reassessment, education on steroid‑induced hyperglycemia.

3. **Integration of COPD and diabetes care**  
   * Coordinate inhaled therapy with glucose control (avoid hypoglycemia when using beta‑agonists).  
   * Use multidisciplinary teams (pulmonology, endocrinology, pharmacy) for medication reconciliation.  
   * Emphasize that GLP‑1 analogues or DPP‑4 inhibitors have no proven direct benefit for COPD symptoms or exacerbation reduction; they should be used for metabolic indications only.
```
</details>

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure you have `ollama` running locally with the `qwen3:4b` model (or configure `OPENROUTER_API_KEY` in your `.env` for remote models).
4.  Place your medical PDF documents in `gepa/docs/`.

## Usage

To run the multi-agent system:

```bash
python MultiAgent.py
```

To run the optimization process:

```bash
python gepa_optimisation.py
```

## Tools & Libraries

*   **DSPy**: For programming and optimizing language model agents.
*   **LangChain**: For document loading, splitting, and vector storage.
*   **FAISS**: For efficient similarity search of medical passages.
*   **MLflow**: For experiment tracking and logging evaluation metrics.

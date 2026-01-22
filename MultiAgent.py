# Load the dataset
with open(SCRIPT_DIR / "docs/qa_pairs_diabets.json", "r") as f:
    qa_copd_data = json.load(f)

# Convert to dspy.Example objects
dataset_copd = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_copd_data]

# shuffle the dataset
random.shuffle(dataset_copd)

# Split the dataset as requested
trainset_copd = dataset_copd[:10]
devset_copd = dataset_copd[10:]

print(f"Loaded {len(dataset_copd)} examples.")
print(f"Train set size: {len(trainset_copd)}")
print(f"Dev set size: {len(devset_copd)}")

class COPDAgent(dspy.Module):
    def __init__(
        self
    ):
        # init LLM - using local Ollama model
        self.lm = dspy.LM(
            model="ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            api_key="",
            temperature=0.3,
            cache=False
        )
        dspy.configure(lm=self.lm)
        self.copd_agent = dspy.ReAct(ReActSignature, tools=[copd_vector_search_tool])

    def forward(self, question: str):
        return self.copd_agent(question=question)
    
copd_agent = COPDAgent()

evaluator_copd = Evaluate(devset=devset_copd, num_threads=32, display_progress=True, display_table=5, provide_traceback=True)

# Evaluate the baseline agent (the existing `react`)
print("Evaluating the baseline ReAct agent...")
copd_baseline_eval = evaluator_copd(copd_agent, metric=llm_metric_prediction)
copd_baseline_eval


copd_agent.copd_agent.extract._compiled = True
copd_agent.copd_agent.react._compiled = False
teleprompter = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=2,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
)

optimized_copd_agent = teleprompter.compile(student=copd_agent, trainset=trainset_copd, valset=devset_copd)


results = optimized_copd_agent.detailed_results

# Get all candidates and their validation scores
candidates = results.candidates

val_scores = results.val_aggregate_scores

# Find the best candidate by validation score
best_idx = results.best_idx  # This is automatically calculated
best_score = val_scores[best_idx]
best_candidate = results.best_candidate

print(f"Best candidate index: {best_idx}")
print(f"Best validation score: {best_score}")
print(f"Best candidate components: {best_candidate}")

optimized_diabetes_agent.save("dspy_program/optimized_react_diabets.json", save_program=False)
optimized_copd_agent.save("dspy_program/optimized_react_copd.json", save_program=False)

def ask_diabetes(question: str) -> str:
    """Call the Diabetes expert agent and return its answer text."""
    pred = optimized_diabetes_agent(question=question)
    return pred.answer


def ask_copd(question: str) -> str:
    """Call the COPD expert agent and return its answer text."""
    pred = optimized_copd_agent(question=question)
    return pred.answer



class LeadReAct(dspy.Module):
    def __init__(self):
        self.lm = dspy.LM(
            "openrouter/openai/gpt-oss-20b", 
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            temperature=0.3, 
            max_tokens=64000,
            cache=False
        )
        dspy.configure(lm=self.lm)
        self.lead_react = dspy.ReAct(ReActSignature, tools=[ask_diabetes, ask_copd])

    def forward(self, question: str):
        return self.lead_react(question=question)
    
lead_react = LeadReAct()

### combining 2 datasets and creatinga new one so that the lead agent uses the new dataset and uses he subagents to answer the questions


with open(SCRIPT_DIR / "docs/qa_pairs_diabets.json", "r") as f:
    qa_joint_data = json.load(f)

# Convert to dspy.Example objects
joint_dataset = [dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question") for item in qa_joint_data]

# shuffle the dataset
random.shuffle(joint_dataset)

# Split the dataset as requested
trainset_joint = joint_dataset[:train_size]
devset_joint = joint_dataset[train_size:]

print(f"Loaded {len(joint_dataset)} examples.")
print(f"Train set size: {len(trainset_joint)}")
print(f"Dev set size: {len(devset_joint)}")


evaluator_joint = Evaluate(devset=devset_joint, num_threads=32, display_progress=True, display_table=5, provide_traceback=True)
print("Evaluating baseline Lead ReAct (agents-as-tools) on joint dev set...")
baseline_lead_eval = evaluator_joint(lead_react, metric=llm_metric_prediction)
baseline_lead_eval


lead_react.lead_react.extract._compiled = True
lead_react.lead_react.react._compiled = False

teleprompter_joint = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=3,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=teacher_lm,
)

optimized_lead_react = teleprompter_joint.compile(student=lead_react, trainset=trainset_joint, valset=devset_joint)


# Access the detailed results from your optimized agent
results = optimized_lead_react.detailed_results

# Get all candidates and their validation scores
candidates = results.candidates

val_scores = results.val_aggregate_scores

# Find the best candidate by validation score
best_idx = results.best_idx  # This is automatically calculated
best_score = val_scores[best_idx]
best_candidate = results.best_candidate

print(f"Best candidate index: {best_idx}")
print(f"Best validation score: {best_score}")
print(f"Best candidate components: {best_candidate}")

optimized_lead_react.save("dspy_program/optimized_lead_react.json", save_program=False)
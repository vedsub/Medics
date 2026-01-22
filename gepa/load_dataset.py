import dspy
import json
import random
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Load the QA pairs
with open(SCRIPT_DIR / "docs/qa_pairs_diabets.json", "r") as f:
    qa_diabetes_data = json.load(f)

# Create DSPy Examples
dataset_diabetes = [
    dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question")
    for item in qa_diabetes_data
]

# Shuffle the dataset
random.shuffle(dataset_diabetes)

# Split the dataset
train_size = 20
trainset_diabetes = dataset_diabetes[:train_size]
devset_diabetes = dataset_diabetes[train_size:]

print(f"Loaded {len(dataset_diabetes)} examples.")
print(f"Train set size: {len(trainset_diabetes)}")
print(f"Dev set size: {len(devset_diabetes)}")

# Show a few examples
print("\n--- Sample Train Examples ---")
for i, ex in enumerate(trainset_diabetes[:3]):
    print(f"\n[{i+1}] Q: {ex.question[:80]}...")
    print(f"    A: {ex.answer[:80]}...")

import sys
from pathlib import Path
import csv

# Find project root and src folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import DP-MLM
from dpmlm import DPMLM

print("DP-MLM imported successfully!")

# Create DP-MLM model
M = DPMLM(MODEL="FacebookAI/roberta-base")

# Read first 10 SST-2 training examples
data_file = PROJECT_ROOT / "data" / "SST-2" / "train.tsv"

texts = []

with open(data_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter="\t")

    for row in reader:
        texts.append({
            "sentence": row["sentence"],
            "label": row["label"]
        })

        if len(texts) == 10:
            break

# Test epsilon values
epsilons = [10, 25, 50, 100, 250]

# Run DP-MLM
for item in texts:

    print("\nOriginal:")
    print(item["sentence"])
    print("Label:", item["label"])

    for epsilon in epsilons:

        private_text, start, end = M.dpmlm_rewrite(
            item["sentence"],
            epsilon=epsilon
        )

        print("\nEpsilon:", epsilon)
        print("Privatized:")
        print(private_text)
        print("Start:", start, "End:", end)

print("\nSST-2 test completed!")
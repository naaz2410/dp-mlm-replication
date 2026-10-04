import sys
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dpmlm import DPMLM

print("Loading DP-MLM...")
M = DPMLM(MODEL="FacebookAI/roberta-base")
print("DP-MLM loaded successfully!")

data_file = PROJECT_ROOT / "data" / "SST-2" / "train.tsv"

texts = []

with open(data_file, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter="\t")

    for row in reader:
        texts.append({
            "sentence": row["sentence"],
            "label": row["label"]
        })

        if len(texts) == 100:
            break

epsilons = [1, 5, 10, 25, 50, 100]

output_file = PROJECT_ROOT / "experiments" / "dpmlm_results_100.csv"

with open(output_file, "w", encoding="utf-8", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "original_text",
        "label",
        "epsilon",
        "privatized_text",
        "start",
        "end"
    ])

    total = len(texts) * len(epsilons)
    count = 0

    for item in texts:

        print("\nOriginal:")
        print(item["sentence"])
        print("Label:", item["label"])

        for epsilon in epsilons:

            count += 1

            print(f"Running {count}/{total} (epsilon={epsilon})")

            try:
                private_text, start, end = M.dpmlm_rewrite(
                    item["sentence"],
                    epsilon=epsilon
                )

                writer.writerow([
                    item["sentence"],
                    item["label"],
                    epsilon,
                    private_text,
                    start,
                    end
                ])

                file.flush()

            except Exception as e:
                print("ERROR:", e)

print("\nExperiment completed!")
print("Results saved to:")
print(output_file)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT = "experiments/dpmlm_results_100.csv"
OUTPUT = "experiments/dpmlm_results_100_classified.csv"
MODEL_NAME = "textattack/roberta-base-SST-2"

print("Loading SST-2 classifier...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

df = pd.read_csv(INPUT)
predictions = []

print("Evaluating", len(df), "privatized sentences...")

texts = df["privatized_text"].fillna("").tolist()

for i in range(0, len(texts), 16):
    batch_texts = texts[i:i+16]

    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    batch_predictions = outputs.logits.argmax(dim=-1).tolist()
    predictions.extend(batch_predictions)

    print(f"Processed {min(i+16, len(texts))}/{len(texts)}")

df["predicted_label"] = predictions
df["correct"] = df["label"].astype(int) == df["predicted_label"]

df.to_csv(OUTPUT, index=False)

print()
print("Classification results saved to:")
print(OUTPUT)

print()
print("Accuracy by epsilon:")
print(df.groupby("epsilon")["correct"].mean())

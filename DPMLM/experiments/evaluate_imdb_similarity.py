import pandas as pd
from sentence_transformers import SentenceTransformer, util

INPUT = "experiments/imdb_dpmlm_results_6.csv"
OUTPUT = "experiments/imdb_dpmlm_results_6_evaluated.csv"

print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(INPUT)

print(f"Evaluating {len(df)} privatized sentences...")

original_embeddings = model.encode(
    df["original_text"].tolist(),
    convert_to_tensor=True
)

privatized_embeddings = model.encode(
    df["privatized_text"].tolist(),
    convert_to_tensor=True
)

similarities = util.cos_sim(
    original_embeddings,
    privatized_embeddings
).diagonal()

df["similarity"] = similarities.cpu().numpy()

df.to_csv(OUTPUT, index=False)

print("\nSimilarity evaluation completed!")
print("Results saved to:")
print(OUTPUT)

print("\nAverage similarity by epsilon:")
print(df.groupby("epsilon")["similarity"].mean())

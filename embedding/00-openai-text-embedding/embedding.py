import os
import openai
import requests
import json
import pandas as pd
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    """Get embedding for text using OpenAI API."""
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def get_model_list(endpoint: str = "v1/models") -> list[str]:
    """Get list of available embedding models."""
    header = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    return requests.get(
        f"https://api.openai.com/{endpoint}",
        headers=header,
    )


# Check embedding model list
model_list = get_model_list()
with open("model_list.json", "w") as f:
    json.dump(model_list.json(), f)

# Get ada002 embedding (Second generation model)
embedding01 = get_embedding("Embedding example", model="text-embedding-ada-002")
print("shape of embedding01 : ", len(embedding01))

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_tokenizer = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


# load & inspect dataset
input_datapath = (
    "data/fine_food_reviews_1k.csv"  # to save space, we provide a pre-filtered dataset
)
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

# subsample to 1k most recent reviews and remove samples that are too long
top_n = 1000
df = df.sort_values("Time").tail(
    top_n * 2
)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)


# get encoding (tokenizer) for embedding model
tokenizer = tiktoken.get_encoding(embedding_tokenizer)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
print(len(df))

# Get embeddings
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model=embedding_model))
df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")

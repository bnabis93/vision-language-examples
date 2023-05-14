import os
import requests
import json


def get_fine_tune_model_list(endpoint: str = "v1/fine-tunes") -> list[str]:
    """Get list of available embedding models."""
    header = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    return requests.get(
        f"https://api.openai.com/{endpoint}",
        headers=header,
    )


# Get fine-tune model list
model_list = get_fine_tune_model_list()
with open("fine_tune_model.json", "w") as f:
    json.dump(model_list.json(), f)

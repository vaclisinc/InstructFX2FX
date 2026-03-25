import re
import json
import numpy as np

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip("` \n")
    return json.loads(text)

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_serializable(v) for v in obj)
    return obj
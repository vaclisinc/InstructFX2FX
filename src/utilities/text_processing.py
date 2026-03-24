import re
import json

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip("` \n")
    return json.loads(text)

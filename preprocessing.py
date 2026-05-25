import re


def clean_text(text):
    # normalize whitespace
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_function(text):
    text = clean_text(text)
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)

import json

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open("vocab.txt", "w", encoding="utf-8") as f:
    for token, idx in vocab.items():
        f.write(f"{token} {idx}\n")
import json

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("docs.jsonl", "w", encoding="utf-8") as f:
    for i, text in enumerate(chunks):
        obj = {"doc_id": str(i), "text": text}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
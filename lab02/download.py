import os
from speakleash import Speakleash

data_dir = "speakleash_data"
os.makedirs(data_dir, exist_ok=True)

DOC_LIMIT = 10_000
OUTPUT_FILE = "corpus.txt"

sl = Speakleash(data_dir)
dataset = sl.get("plwiki").data
assert dataset is not None

count = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for doc in dataset:
        if count >= DOC_LIMIT:
            break
        if text := doc.strip():
            f.write(text + "\n")
            count += 1

print(f"Rozmiar pliku: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")

with open("corpus.txt", encoding="utf-8") as file:
    text = file.read()

text_train = text[: int(len(text) * 0.9)]
text_valid = text[int(len(text) * 0.9) :]

with open("train.txt", "w", encoding="utf-8") as file:
    file.write(text_train)
    
with open("valid.txt", "w", encoding="utf-8") as file:
    file.write(text_valid)

os.remove("corpus.txt")

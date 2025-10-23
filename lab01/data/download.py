import os
import re
import requests


url = "https://wolnelektury.pl/api/authors/henryk-sienkiewicz/books/"
fpath = os.path.join(os.path.dirname(__file__), "sienkiewicz.txt")
books = requests.get(url).json()

with open(fpath, "w", encoding="utf-8") as file:
    for book in books:
        try:
            book_url = book["href"]
            book_json = requests.get(book_url).json()

            text_url = book_json["txt"]
            text = requests.get(text_url).text

            file.write(text)
            file.write("\n\n\n")

        except Exception as e:
            print(f"Error: {e} while getting {book['title']}")

with open(fpath, "r", encoding="utf-8") as file:
    text = file.read()

pattern = r"-{5,}.*?(\n\s*Henryk Sienkiewicz\s*\n)"
replacement = r"\1"
clean_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

with open(fpath, "w", encoding="utf-8") as file:
    file.write(clean_text)

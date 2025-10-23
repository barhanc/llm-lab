import os
import requests


authors = requests.get("https://wolnelektury.pl/api/authors").json()
fpath = os.path.join(os.path.dirname(__file__), "wolne_lektury.txt")

with open(fpath, "w", encoding="utf-8") as file:
    for url in map(lambda d: d.get("href"), authors):
        print(url)
        books = requests.get(url + "books/").json()
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

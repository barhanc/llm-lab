from collections import Counter
from itertools import pairwise
from math import inf
from tqdm import trange


class Tokenizer:
    def __init__(self, characters: list[str]):
        self.characters = characters
        self._stoi = {ch: i for i, ch in enumerate(characters)}
        self._itos = {i: ch for i, ch in enumerate(characters)}
        self._merges: dict[tuple[int, int], int] = {}

    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_tok: int) -> list[int]:
        i, t = 0, []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                t.append(new_tok)
                i += 2
            else:
                t.append(tokens[i])
                i += 1
        return t

    @staticmethod
    def _get_pair_frequency(tokens: list[int]) -> Counter[tuple[int, int]]:
        return Counter(pairwise(tokens))

    def fit(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < len(self.characters):
            raise ValueError(f"{vocab_size=} must be >= than {len(self.characters)=}")
        if set(text) != set(self._stoi):
            raise ValueError("Text contains unrecognized character")

        tokens = [self._stoi[ch] for ch in text]

        for i in trange(vocab_size - len(self.characters), disable=not verbose):
            # --- Get the most frequent pair of tokens ---
            freq = self._get_pair_frequency(tokens)
            pair = freq.most_common(n=1)[0][0]
            # --- Mint new token ---
            new_tok = len(self.characters) + i
            # --- Replace every occurence of `pair` with `new_tok` ---
            tokens = self._merge(tokens, pair, new_tok)
            # --- Update mappings ---
            self._merges[pair] = new_tok
            self._itos[new_tok] = self._itos[pair[0]] + self._itos[pair[1]]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._itos[tok] for tok in tokens)

    def encode(self, text: str) -> list[int]:
        tokens = [self._stoi[ch] for ch in text]

        while len(tokens) >= 2:
            freq = self._get_pair_frequency(tokens)
            pair = min(freq, key=lambda p: self._merges.get(p, +inf))
            if pair not in self._merges:
                break
            tokens = self._merge(tokens, pair, self._merges[pair])

        return tokens


if __name__ == "__main__":
    with open("data/mickiewicz.txt", encoding="utf-8") as file:
        text = file.read()

    characters = sorted(set(text))
    tokenizer = Tokenizer(characters)
    tokenizer.fit(text, vocab_size=256, verbose=True)

    assert text == tokenizer.decode(tokenizer.encode(text))

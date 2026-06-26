from .base import merge, get_stats, Tokenizer


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        # Given a string, create a vocabulary
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)  # Copy so we don't destroy the original list

        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)  # Find most commonly occuring pair
            idx = 256 + i  # Mint a new token for it
            print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)  # Replace pairs with new token. 
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]  # Concatenates

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab
        

    def encode(self, text):
        # Given a string, return list of integers (the tokens)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))  # Finds the key (pair) in the stats dictionary that comes earliest in the merges dictionary
            if pair not in self.merges:
                break  # When no mergable pairs remain (first float("inf"))
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    def decode(self, ids):
        # Given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)  # concat into bytes (should perfectly undo encode)
        text = tokens.decode("utf-8", errors='replace')  # not every bit string is valid in utf-8 -> �
        return text


def test():
    text = ""
    with open("tests/taylorswift.txt") as file:
        text = file.read()
    
    tokenizer = BasicTokenizer()
    tokenizer.train(text, 1000, verbose=True)

    msg = "The quick brown fox jumps over the lazy dog. And also 2 + 2 = 4.✨✨✨"
    print(tokenizer.encode(msg))
    print(len(tokenizer.encode(msg)))
    print(tokenizer.decode(tokenizer.encode(msg)))

if __name__ == "__main__":
    test()
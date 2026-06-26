import regex as re
from .base import merge, get_stats, Tokenizer

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    
    def train(self, text, vocab_size, verbose=False):
        # Given a string, create a vocabulary
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Operate across chunks to prevent overmerging common phrases
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = list(ch.encode("utf-8") for ch in text_chunks)

        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            # Count appearances of each consecutive pair, but not across chunks
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)  # Update stats in place for each chunk

            pair = max(stats, key=stats.get)  # Find most commonly occuring pair
            idx = 256 + i  # Mint a new token for it
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]  # Replace pairs with new token. 
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]  # Concatenates

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab
        
    def register_special_tokens(self, special_tokens):
        # example: {"<|endoftext|>": 100257} (str->int)
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}  # (int->str)

    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # Separate special from not-special into chunks
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)  # capturing groups () preserves all text
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_ordinary(self, text):
        # Encoding for chunks without special tokens
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
        
    def _encode_chunk(self, text_bytes):
        # Given a chunk, return the token ids
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))  # Finds the key (pair) in the stats dictionary that comes earliest in the merges dictionary
            if pair not in self.merges:
                break  # When no mergable pairs remain (occurs when "inf")
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        # Given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors='replace')  # not every bit string is valid in utf-8 -> �
        return text

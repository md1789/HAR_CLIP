import re

class SimpleTokenizer:
    """
    Extremely simplified tokenizer for AlphaCLIP stub.
    Replace with the real tokenizer from the AlphaCLIP repo.
    """
    def __init__(self):
        # Just a toy vocab for testing
        self.encoder = {
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
        }
        self.next_id = 2

    def encode(self, text: str):
        # Very naive: split on spaces, assign new IDs as needed
        tokens = []
        for word in re.findall(r"\w+", text.lower()):
            if word not in self.encoder:
                self.encoder[word] = self.next_id
                self.next_id += 1
            tokens.append(self.encoder[word])
        return tokens

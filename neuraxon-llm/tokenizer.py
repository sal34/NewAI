import json
from typing import List, Dict, Set

class Tokenizer:
    """
    A simple character-level tokenizer to convert text to and from trinary format.
    """
    def __init__(self, corpus: List[str]):
        """
        Initializes the tokenizer and builds the vocabulary.

        Args:
            corpus: A list of strings to build the vocabulary from.
        """
        self.char_to_int: Dict[str, int] = {}
        self.int_to_char: Dict[int, str] = {}
        self.vocab: Set[str] = set()
        self.vocab_size: int = 0
        self._build_vocab(corpus)

    def _build_vocab(self, corpus: List[str]):
        """
        Builds the vocabulary from a corpus of text.

        Args:
            corpus: A list of strings.
        """
        for text in corpus:
            self.vocab.update(text)

        self.vocab = sorted(list(self.vocab))
        self.vocab_size = len(self.vocab)

        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}
        self.int_to_char = {i: char for i, char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """
        Converts a string of text to a list of integers.

        Args:
            text: The input string.

        Returns:
            A list of integers representing the text.
        """
        return [self.char_to_int.get(char, -1) for char in text]

    def decode(self, int_list: List[int]) -> str:
        """
        Converts a list of integers back to a string of text.

        Args:
            int_list: A list of integers.

        Returns:
            The decoded string.
        """
        return "".join([self.int_to_char.get(i, "") for i in int_list])

    def int_to_trinary_vector(self, char_int: int) -> List[int]:
        """
        Converts an integer to a trinary vector (one-hot style).

        Args:
            char_int: The integer representation of a character.

        Returns:
            A trinary vector.
        """
        vector = [0] * self.vocab_size
        if 0 <= char_int < self.vocab_size:
            vector[char_int] = 1
        return vector

    def trinary_vector_to_int(self, vector: List[int]) -> int:
        """
        Converts a trinary vector back to an integer.

        Args:
            vector: A trinary vector.

        Returns:
            The integer representation of the character.
        """
        try:
            return vector.index(1)
        except ValueError:
            # If no 1 is found, find the index of the max value
            try:
                max_val = max(vector)
                if max_val > 0:
                    return vector.index(max_val)
            except ValueError:
                return -1 # Return -1 if the vector is empty
        return -1

    def save(self, filepath: str):
        """
        Saves the tokenizer's vocabulary to a file.
        """
        with open(filepath, 'w') as f:
            json.dump({
                'char_to_int': self.char_to_int,
                'int_to_char': self.int_to_char,
                'vocab': list(self.vocab)
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a tokenizer from a file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create a dummy tokenizer and then populate it
        tokenizer = cls(corpus=[])
        tokenizer.char_to_int = data['char_to_int']
        # Convert string keys back to int keys
        tokenizer.int_to_char = {int(k): v for k, v in data['int_to_char'].items()}
        tokenizer.vocab = set(data['vocab'])
        tokenizer.vocab_size = len(tokenizer.vocab)
        return tokenizer

if __name__ == '__main__':
    # Example Usage
    sample_corpus = ["hello world", "this is a test"]

    # Create and train the tokenizer
    tokenizer = Tokenizer(sample_corpus)
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {''.join(tokenizer.vocab)}")

    # Encode text
    text = "hello"
    encoded = tokenizer.encode(text)
    print(f"Encoded '{text}': {encoded}")

    # Decode integers
    decoded = tokenizer.decode(encoded)
    print(f"Decoded {encoded}: '{decoded}'")

    # Convert to trinary vector
    char_int = tokenizer.char_to_int['h']
    trinary_vector = tokenizer.int_to_trinary_vector(char_int)
    print(f"Trinary vector for 'h': {trinary_vector}")

    # Convert trinary vector back to int
    back_to_int = tokenizer.trinary_vector_to_int(trinary_vector)
    print(f"Integer from trinary vector: {back_to_int}")

    # Save and load
    tokenizer.save("tokenizer.json")
    loaded_tokenizer = Tokenizer.load("tokenizer.json")
    print(f"Loaded vocabulary size: {loaded_tokenizer.vocab_size}")

    # Verify loaded tokenizer
    text = "world"
    encoded = loaded_tokenizer.encode(text)
    print(f"Encoded '{text}' with loaded tokenizer: {encoded}")
    decoded = loaded_tokenizer.decode(encoded)
    print(f"Decoded {encoded} with loaded tokenizer: '{decoded}'")

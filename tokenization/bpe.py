from typing import List

import tiktoken


import tiktoken

class BPE:
    """
    Byte Pair Encoding (BPE) tokenizer wrapper using the tiktoken library.

    This class provides a convenient interface for tokenizing and detokenizing
    text using OpenAI's tiktoken library, which implements efficient BPE
    tokenization similar to GPT models.

    Parameters
    ----------
    encoding_method : str, optional, default: "gpt2"
        The pre-trained tokenizer to use. Must be one of the encoding names
        supported by tiktoken. Common options include:
        - "gpt2": GPT-2 tokenizer (vocabulary size: 50257)
        - "gpt-3.5-turbo": GPT-3.5/ChatGPT tokenizer
        - "gpt-4": GPT-4 tokenizer
        - "cl100k_base": Base tokenizer for GPT-3.5 and GPT-4
        - "p50k_base": Base tokenizer for Codex models
        - "r50k_base": Base tokenizer for GPT-3

    Attributes
    ----------
    tokenizer : tiktoken.Encoding
        The underlying tiktoken Encoding object that performs the actual
        tokenization and detokenization operations.

    Raises
    ------
    ValueError
        If the specified `encoding_method` is not in the list of supported
        encodings provided by tiktoken.

    Examples
    --------
    >>> tokenizer = BPE(encoding_method="gpt2")
    >>> tokens = tokenizer.encode("Hello, world!")
    >>> print(f"Token IDs: {tokens}")
    >>> text = tokenizer.decode(tokens)
    >>> print(f"Decoded text: {text}")

    >>> # Using a different encoding
    >>> chat_tokenizer = BPE(encoding_method="gpt-3.5-turbo")
    >>> tokens = chat_tokenizer.encode("Hello, how can I help you?")

    Notes
    -----
    - The BPE algorithm merges the most frequent pairs of bytes/characters
      iteratively to build a vocabulary of subword units.
    - Different encoding methods have different vocabulary sizes and are
      optimized for different types of text (general text, code, etc.).
    - The "<|endoftext|>" special token is always allowed during encoding
      to handle end-of-text boundaries.
    - tiktoken is a fast BPE tokenizer implemented in Rust, providing
      significant performance benefits over pure Python implementations.

    See Also
    --------
    tiktoken.get_encoding : Function to retrieve a tiktoken Encoding object.
    tiktoken.list_encoding_names : Function to list all available encodings.
    """
    
    def __init__(self, encoding_method: str = "gpt2"):
        """Initialize the BPE tokenizer with the specified encoding method.

        Args:
            encoding_method (str, optional): the tokenization method. Defaults to "gpt2".
        """
        if encoding_method not in tiktoken.list_encoding_names():
            raise ValueError(f"Encoding method {encoding_method} is not supported. "
                             f"Choose from {tiktoken.list_encoding_names()}")
        self.tokenizer = tiktoken.get_encoding(encoding_method)

    def encode(self, text: str):
        """
        Convert input text into a list of token IDs.

        Parameters
        ----------
        text : str
            The input text to tokenize. Can be of any length, but very long
            texts may be truncated by models with context length limits.

        Returns
        -------
        list[int]
            A list of integer token IDs representing the tokenized text.
            The length of the list depends on the tokenizer's vocabulary
            and the complexity of the input text.

        Examples
        --------
        >>> tokenizer = BPE()
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(f"Number of tokens: {len(tokens)}")
        >>> print(f"Token IDs: {tokens}")

        Notes
        -----
        - The method automatically handles special tokens, with only
          "<|endoftext|>" explicitly allowed as a special token.
        - The tokenization is deterministic: the same input text will
          always produce the same token IDs.
        - Whitespace and punctuation are tokenized according to the
          specific BPE merges of the chosen encoding method.
        """
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list[int]):
        """
        Convert a list of token IDs back into text.

        Parameters
        ----------
        token_ids : list[int]
            A list of integer token IDs to decode. The IDs must be valid
            for the tokenizer's vocabulary. Invalid IDs may cause errors
            or produce unexpected output.

        Returns
        -------
        str
            The decoded text reconstructed from the token IDs. For most
            tokenizers, this should be the original text (except for
            minor whitespace differences in some cases).

        Raises
        ------
        ValueError
            If any token ID is outside the valid range for the tokenizer's
            vocabulary (handled internally by tiktoken).

        Examples
        --------
        >>> tokenizer = BPE()
        >>> text = "Hello, world!"
        >>> tokens = tokenizer.encode(text)
        >>> decoded = tokenizer.decode(tokens)
        >>> print(f"Original: '{text}'")
        >>> print(f"Decoded: '{decoded}'")
        >>> assert text == decoded  # Should be true for most cases

        Notes
        -----
        - Decoding is the inverse operation of encoding: 
          `decode(encode(text)) == text` should hold true for most inputs.
        - Some tokenizers may normalize text during tokenization (e.g.,
          collapsing multiple spaces), which can result in minor differences
          between original and decoded text.
        - The method handles special tokens appropriately, converting them
          to their string representations or omitting them as needed.
        """
        return self.tokenizer.decode(token_ids)


if __name__ == "__main__":
    bpe = BPE(encoding_method="gpt2")

    text = "Hello Yazdan! <|endoftext|>"
    print("Original text:")
    print(text)

    token_ids = bpe.encode(text)
    print("\nEncoded token IDs:")
    print(token_ids)

    decoded_text = bpe.decode(token_ids)
    print("\nDecoded text:")
    print(decoded_text)


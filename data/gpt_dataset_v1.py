import torch
from torch.utils.data import Dataset, DataLoader

from tokenization.bpe import BPE

import torch
from torch.utils.data import Dataset, DataLoader
from tokenization.bpe import BPE

class GPTDatasetV1(Dataset):
    """
    PyTorch Dataset for training autoregressive (GPT-style) language models.

    This dataset tokenizes an entire input text and creates overlapping
    input–target token sequences using a sliding window approach.
    Each input sequence is paired with a target sequence that is shifted
    by one token, enabling next-token prediction.

    Parameters
    ----------
    txt : str
        The full raw text corpus used to create training samples.
        No default value.

    tokenizer : Any
        Tokenizer object providing an `encode(text: str) -> List[int]`
        method to convert text into token IDs.
        No default value.

    max_length : int, optional, default: 256
        Length of each input token sequence (context window size).
        Determines the number of tokens the model sees in each training sample.
        Typical values range from 128 to 2048 depending on model architecture
        and memory constraints.

    stride : int, optional, default: 128
        Step size between consecutive sliding windows.
        Smaller values create more overlapping samples (increases dataset size
        but may cause overfitting due to redundant data). Larger values create
        less overlap (reduces dataset size but may lose contextual relationships).
        Must be less than or equal to `max_length`.

    Attributes
    ----------
    input_ids : List[torch.Tensor]
        List of tensors containing input token sequences of shape `(max_length,)`.
        Each tensor represents a context window for the model to process.

    target_ids : List[torch.Tensor]
        List of tensors containing target token sequences of shape `(max_length,)`,
        where each sequence is the corresponding input shifted by one token.
        These are the "labels" for next-token prediction.

    Examples
    --------
    >>> from tokenization.bpe import BPE
    >>> tokenizer = BPE()
    >>> tokenizer.load("path/to/vocab.json")
    >>> with open("corpus.txt", "r") as f:
    >>>     text = f.read()
    >>> dataset = GPTDatasetV1(text, tokenizer, max_length=256, stride=128)
    >>> print(f"Number of samples: {len(dataset)}")
    >>> input_seq, target_seq = dataset[0]
    >>> print(f"Input shape: {input_seq.shape}, Target shape: {target_seq.shape}")

    Notes
    -----
    - The dataset uses a sliding window approach which may result in overlapping
      samples when `stride` < `max_length`.
    - The final window that would exceed the token sequence length is discarded.
    - Target sequences are always exactly one token ahead of their corresponding
      input sequences, enabling autoregressive language modeling.
    - Debug prints for input and target IDs are included in the constructor.
    """

    def __init__(self, txt: str, tokenizer: BPE, max_length: int = 256, stride: int = 128):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        print(f"**********input IDs: {self.input_ids}")
        print(f"**********target IDs: {self.target_ids}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            The number of input-target pairs generated from the text corpus
            using the sliding window approach with the specified `max_length`
            and `stride` parameters.

        Notes
        -----
        The count is calculated as:
        `floor((total_tokens - max_length) / stride) + 1` when stride evenly divides
        the remaining tokens, or simply the number of windows that can be extracted
        without exceeding the token sequence bounds.
        """
        return len(self.input_ids)

    def __getitem__(self, index):
        """
        Retrieves the input-target pair at the specified index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve. Must be in the range
            `[0, len(dataset) - 1]`. If using a DataLoader with shuffling,
            indices will be provided in random order.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - input_ids : torch.Tensor
                Token IDs for the input sequence, shape `(max_length,)`
            - target_ids : torch.Tensor
                Token IDs for the target sequence, shape `(max_length,)`
        
        Raises
        ------
        IndexError
            If `index` is out of the valid range `[0, len(self) - 1]`.

        Examples
        --------
        >>> dataset = GPTDatasetV1(text, tokenizer, max_length=10, stride=5)
        >>> input_seq, target_seq = dataset[0]
        >>> print(f"Input tokens: {input_seq}")
        >>> print(f"Target tokens: {target_seq}")
        >>> # The target sequence should be the input sequence shifted by 1 token
        >>> assert torch.all(target_seq[:-1] == input_seq[1:])

        Notes
        -----
        - Both returned tensors are of dtype `torch.long` by default.
        - No additional preprocessing (like padding) is applied since all
          sequences are guaranteed to have length `max_length`.
        - This method is automatically called by PyTorch's DataLoader during
          iteration.
        """
        return self.input_ids[index], self.target_ids[index]


if __name__ == "__main__":
    print("fffffffffffffffffffffffff")
    txt = "We will make the Iran great again! we will make our country the heaven on earth! we will make our people happy and prosperous! we will make our economy strong and stable! we will make our culture rich and vibrant! we will make our future bright and hopeful! we will make our dreams come true! we will make our vision a reality! we will make our mission successful! we will make our goals achievable! we will make our plans effective! we will make our actions impactful! we will make our efforts fruitful! we will make our work meaningful! we will make our lives fulfilling! we will make our legacy enduring!")
    print(f"fffffffffffffffffffffffff")
    # Simple test text
    # txt = "Hello world! This is a test sentence for the GPT dataset."
    print(f"*******************text: {txt}")
    # Import and create tokenizer
    from tokenization.bpe import BPE
    tokenizer = BPE()
    
    # Create dataset
    dataset = GPTDatasetV1(
        txt=txt,
        tokenizer=tokenizer,
        max_length=8,
        stride=4
    )
    
    
    
    print(f"=====================================================")
    # Basic information
    print(f"Text length: {len(txt)} characters")
    print(f"Number of tokens: {len(tokenizer.encode(txt))}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Max sequence length: 8")
    print(f"Stride: 4")
    
    # Show first few samples
    print("\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        input_seq, target_seq = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Input:  {input_seq.tolist()}")
        print(f"  Target: {target_seq.tolist()}")
        print(f"  Input text:  '{tokenizer.decode(input_seq.tolist())}'")
        print(f"  Target text: '{tokenizer.decode(target_seq.tolist())}'")

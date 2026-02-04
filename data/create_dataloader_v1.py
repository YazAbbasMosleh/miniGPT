from torch.utils.data import DataLoader

from tokenization.bpe import BPE
from data.gpt_dataset_v1 import GPTDatasetV1




class DataLoader:
    """
    DataLoader combines a dataset and a sampler, and provides an iterable 
    over the given dataset. It enables efficient batching, shuffling, and 
    data loading for training machine learning models.
    
    This implementation supports mini-batch training where data is loaded 
    in batches rather than individually, improving computational efficiency 
    through vectorized operations.
    
    Parameters
    ----------
    dataset : GPTDatasetV1
        The dataset object containing the data to be loaded. Must implement 
        the necessary methods for accessing data samples.
        
    batch_size : int, optional, default: 4
        Number of samples per batch to load. A typical batch size ranges 
        from 16 to 256 depending on available memory and model requirements.
        Smaller batch sizes provide more frequent weight updates but less 
        stable gradient estimates, while larger batches offer more stable 
        gradients but require more memory.
        
    shuffle : bool, optional, default: True
        If True, the data is reshuffled at every epoch. Shuffling helps 
        prevent the model from learning the order of training examples and 
        reduces overfitting. It's typically set to True for training data 
        and False for validation/test data.
        
    drop_last : bool, optional, default: True
        If True, drops the last incomplete batch when the dataset size is 
        not divisible by the batch size. This ensures all batches have the 
        same size, which can be important for certain operations like 
        batch normalization. If False, the last batch will be smaller than 
        the specified batch size.
        
    Attributes
    ----------
    dataset : GPTDatasetV1
        The underlying dataset object.
        
    batch_size : int
        The number of samples in each batch.
        
    shuffle : bool
        Whether to shuffle the data at each epoch.
        
    drop_last : bool
        Whether to drop the last incomplete batch.
        
    Examples
    --------
    >>> from dataset import GPTDatasetV1
    >>> dataset = GPTDatasetV1(data, tokenizer, max_length=512)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    >>> for batch in dataloader:
    >>>     # Process batch
    >>>     inputs, targets = batch
    >>>     outputs = model(inputs)
    
    Notes
    -----
    - The DataLoader uses a sampler internally to handle shuffling and batching.
    - When `drop_last=True`, some data may not be used in each epoch if the 
      dataset size is not divisible by the batch size.
    - For multi-GPU training, consider using distributed data parallel 
      versions with appropriate sampling strategies.
    
    See Also
    --------
    torch.utils.data.DataLoader : PyTorch's native DataLoader implementation.
    """
    
    def __init__(self, dataset: GPTDatasetV1, batch_size: int = 4, 
                 shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def create_dataloader_v1(self):
        """
        Creates and returns a DataLoader instance configured with the 
        current parameters.
        
        Returns
        -------
        DataLoader
            A DataLoader instance ready for iterating over batches of data.
            
        Note
        ----
        This method is a factory method that returns a new DataLoader 
        rather than modifying the current instance in-place. Consider 
        making the current class itself iterable for a more intuitive API.
        """
        dataLoader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                drop_last=self.drop_last)
        return dataLoader


if __name__ == "__main__":
    # from torch.utils.data import Dataset, 
    
    from data.gpt_dataset_v1 import GPTDatasetV1
    from tokenization.bpe import BPE
    
    with open("./verdict/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    tokenizer = BPE()
    dataset = GPTDatasetV1(txt=raw_text, tokenizer=tokenizer, stride=1, max_length=4)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    dataloader = dataloader.create_dataloader_v1()
    print(dataloader)
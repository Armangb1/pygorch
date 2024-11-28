from abc import ABC
import random

class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    

class TensorDataset(Dataset):
   
    def __init__(self, *tensors) -> None:
        super().__init__()
        self.tensors = tensors

    def __getitem__(self, idx):
        items = ()
        for tensor in self.tensors:
            item = tensor[idx]
            items = items + (item,)
        return items
    
    def __len__(self):
        tensors = self.tensors
        return len(tensors[0].value)

class DataLoader:
    def __init__(self, dataset, batchsize, shuffle=False) -> None:
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        idx = self.idx
        dataset = self.dataset
        batchsize = self.batchsize

        if idx >= len(dataset):
            raise StopIteration
        nextIdx = idx + batchsize
        if nextIdx >= len(dataset):
            nextIdx = len(dataset)
        batch_indices = self.indices[idx:nextIdx]
        out = dataset[batch_indices]
        self.idx = nextIdx
        return out
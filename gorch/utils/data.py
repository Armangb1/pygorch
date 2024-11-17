from abc import ABC

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
    def __init__(self, dataset, batchsize) -> None:
        self.dataset = dataset
        self.batchsize = batchsize

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        idx = self.idx
        dataset = self.dataset
        batchsize = self.batchsize

        if idx== len(dataset):
            raise StopIteration
        nextIdx = idx+batchsize
        if nextIdx>=len(dataset):
            nextIdx = len(dataset)
        out = dataset[idx:nextIdx]
        self.idx = nextIdx
        return out
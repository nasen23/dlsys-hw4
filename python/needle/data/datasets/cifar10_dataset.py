import os
import pickle
from pathlib import Path
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        base = Path(base_folder)
        if train:
            data = []
            labels = []
            for i in range(1, 6):
                with (base / f"data_batch_{i}").open("rb") as f:
                    d = pickle.load(f, encoding="bytes")
                data.append(d[b"data"])
                labels.extend(d[b"labels"])
            self.data = np.reshape(data, (50000, 3, 32, 32)).astype(np.float32) / 255
            self.labels = np.array(labels)
        else:
            with (base / "test_batch").open("rb") as f:
                d = pickle.load(f, encoding="bytes")
            self.data = np.reshape(d[b"data"], (10000, 3, 32, 32)).astype(np.float32) / 255
            self.labels = np.array(d[b"labels"])
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        x, y = self.data[index], self.labels[index]
        if self.transforms:
            for transform in self.transforms:
                x = transform(x)
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.data.shape[0]
        ### END YOUR SOLUTION

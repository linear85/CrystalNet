import numpy as np
import pickle
import torch
from typing import Union, List, Tuple, Optional
from torch_geometric.data.data import Data
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import random

class LoadData:
    def __init__(self, data_src: str) -> None:
        self.data_type = False
        self.data = self.readData(data_src)

    def KFold(self, fold=10, batch_size=1, random_seed=7268128) -> List[Tuple[DataLoader]]:
        kFold_data = []
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_seed)
        for (train_index, test_index) in kf.split(self.data):
            if self.data_type == 'np':
                cur_train_fold = self.np2DataLoader(self.data[train_index], batch_size)
                cur_test_fold  = self.np2DataLoader(self.data[test_index], batch_size)
            else:
                cur_train_fold = DataLoader(self.data, sampler=train_index, batch_size=batch_size, shuffle=False)
                cur_test_fold  = DataLoader(self.data, sampler=test_index, batch_size=batch_size, shuffle=False)
            kFold_data.append((cur_train_fold, cur_test_fold))
        return kFold_data
    
    @staticmethod
    def np2DataLoader(data: np.ndarray, batch_size) -> DataLoader:
        x, y = torch.Tensor(data[:, :-1]), torch.Tensor(data[:, -1:])
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def readData(self, data_src: str) -> Union[np.ndarray, List[Data]]:
        print("Read data!")
        if "npy" in data_src:
            print("np_ndarray data format")
            self.data_type = 'np'
            data = self.load_NP_ARRAY(data_src)
        else:
            print("list of gnn data format")
            self.data_type = 'gnn'
            data = self.load_Pickled_List(data_src)
        return data
    
    @staticmethod
    def load_NP_ARRAY(data_src: str) -> np.ndarray:
        data = np.load(data_src)
        return data
    
    @staticmethod
    def load_Pickled_List(data_src: str) -> List[Data]:
        open_file   = open(data_src, "rb")
        data = pickle.load(open_file)
        open_file.close()
        return data
    
    def transformInput(self, partition: float = 0.8, train_batch_size: int = 1, val_batch_size: int = 1, shuffle: bool = True, random_see: int = 66) -> Tuple[DataLoader, Optional[DataLoader]]:
        data_size    = len(self.data)
        random.seed(random_see)
        if (shuffle):
            random.shuffle(self.data)
        if partition == -1:
            train_loader = DataLoader(self.data, batch_size=train_batch_size)
            return train_loader, None
        train_loader       = DataLoader(self.data[:int(data_size*partition)], batch_size=train_batch_size)
        validation_loader  = DataLoader(self.data[int(data_size*partition):], batch_size=val_batch_size)
        return train_loader, validation_loader
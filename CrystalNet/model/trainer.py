import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

def trainMode(model: torch.nn.Module,
              train_data: DataLoader, 
              test_data: DataLoader, 
              device = 'cuda', 
              loss_fn_train = torch.nn.MSELoss(), 
              loss_fn_test  = torch.nn.L1Loss(),
              optimizer = torch.optim.SGD,
              lr = 0.0001,
              epoch = 500,
              num_atom = 54,
              seed = 4512151) -> list[list[float]]:
    torch.manual_seed(seed)
    cur_optimizer = optimizer(model.parameters(), lr=lr)
    model.to(device)
    res = []
    for i in range(epoch):
        train_loss_list = []
        for batch in tqdm(train_data):
            batch.to(device)
            pred = model(batch)
            loss = loss_fn_train(pred, batch.y_sum)
            cur_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            cur_optimizer.step()
            train_loss_list.append(np.sqrt(loss.item()))
        train_mean_loss = np.mean(train_loss_list)/num_atom
        test_loss_list = []
        for batch in tqdm(test_data):
            batch.to(device)
            pred = model(batch)
            loss = loss_fn_test(pred, batch.y_sum)
            test_loss_list.append(loss.item())
        test_mean_loss = np.mean(test_loss_list)/num_atom

        print(f"Epoch at {i}, training_loss is {train_mean_loss}, test_loss is {test_mean_loss}")
        res.append([train_mean_loss, test_mean_loss])
    return res


class trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 device = 'cuda', 
                 optimizer = torch.optim.SGD, 
                 lr = 0.0001, 
                 seed = 4512151) -> None:
        self.model = model
        self.device = device
        torch.manual_seed(seed)
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.model.to(device)
    
    def train(self, 
              train_data: DataLoader, 
              validate_data: DataLoader, 
              model_path: str,
              loss_path: str,
              epoch: int = 500, 
              save_model_criteria: tuple[float] = None,
              save_model = False,
              loss_fn_train = torch.nn.MSELoss(), 
              loss_fn_validation = torch.nn.L1Loss()) -> None:
        losses = []
        for i in range(epoch):
            train_loss = self.update(train_data, loss_fn_train)
            validate_loss = self.validate(validate_data, loss_fn_validation)
            print(f"Epoch at {i}, training_loss is {train_loss}, test_loss is {validate_loss}")
            losses.append([train_loss, validate_loss])
            if i%500 == 0:
                self.savePerformance(losses, loss_path)
            if save_model_criteria:
                if train_loss < save_model_criteria[0] and validate_loss < save_model_criteria[1]:
                    self.saveModel(model_path)
                    self.savePerformance(losses, loss_path)
                    return

        self.savePerformance(losses, loss_path)
        if save_model:
            self.saveModel(model_path)
    
    def update(self, data: DataLoader, loss_fn = torch.nn.MSELoss(), clip=1) -> float:
        losses = []
        for batch in tqdm(data):
            batch.to(self.device)
            pred = self.model(batch)
            loss = loss_fn(pred, batch.y_sum)
            self.optimizer.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            losses.append(np.sqrt(loss.item()))
        return np.mean(losses)

    def validate(self, data: DataLoader, loss_fn = torch.nn.L1Loss()) -> float:
        losses = []
        for batch in tqdm(data):
            batch.to(self.device)
            pred = self.model(batch)
            loss = loss_fn(pred, batch.y_sum)
            losses.append(loss.item())
        return np.mean(losses)
    
    def saveModel(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    @staticmethod
    def savePerformance(losses: list[float], path: str) -> None:
        pd.DataFrame(losses).to_excel(path)

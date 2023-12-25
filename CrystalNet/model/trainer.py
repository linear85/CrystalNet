import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd



class trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 level = "sample",
                 device = 'cuda', 
                 optimizer = torch.optim.SGD, 
                 lr = 0.0001, 
                 seed = 4512151) -> None:
        torch.manual_seed(seed)
        self.model = model
        assert level in ("sample", "node"), "the level should be sample or node"
        self.level = level
        self.device = device
        torch.manual_seed(seed)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.model.to(device)
    
    def train(self, 
              train_data: DataLoader, 
              validate_data: DataLoader, 
              loss_path: str,
              clip: float = 10,
              model_path: str = None,
              epoch: int = 500, 
              save_model_criteria: tuple[float] = None,
              loss_fn_train = torch.nn.MSELoss(), 
              loss_fn_validation = torch.nn.L1Loss()) -> None:
        losses = []
        for i in range(epoch):
            train_loss = self.update(train_data, clip, loss_fn_train)
            validate_loss = self.validate(validate_data, loss_fn_validation)
            print(f"Epoch at {i}, training_loss is {train_loss}, validate_loss is {validate_loss}")
            losses.append([train_loss, validate_loss])
            if i%500 == 0:
                self.savePerformance(losses, loss_path)
            if save_model_criteria:
                if train_loss < save_model_criteria[0] and validate_loss < save_model_criteria[1]:
                    self.saveModel(model_path)
                    self.savePerformance(losses, loss_path)
                    return

        self.savePerformance(losses, loss_path)
        if save_model_criteria:
            self.saveModel(model_path)
    
    def trainTest(self, 
              train_data: DataLoader, 
              validate_data: DataLoader, 
              test_data: DataLoader,
              loss_path: str,
              clip: float = 10,
              model_path: str = None,
              epoch: int = 500, 
              save_model_criteria: tuple[float] = None,
              loss_fn_train = torch.nn.MSELoss(), 
              loss_fn_validation = torch.nn.L1Loss()) -> None:
        losses = []
        for i in range(epoch):
            train_loss = self.update(train_data, clip, loss_fn_train)
            validate_loss = self.validate(validate_data, loss_fn_validation)
            test_loss = self.validate(test_data, loss_fn_validation)
            print(f"Epoch at {i}, training_loss is {train_loss}, validate_loss is {validate_loss}, test_loss in {test_loss}")
            losses.append([train_loss, validate_loss, test_loss])
            if i%500 == 0:
                self.savePerformance(losses, loss_path)
            if save_model_criteria:
                if train_loss < save_model_criteria[0] and validate_loss < save_model_criteria[1] and test_loss < save_model_criteria[2]:
                    self.saveModel(model_path)
                    self.savePerformance(losses, loss_path)
                    return

        self.savePerformance(losses, loss_path)
        if save_model_criteria:
            self.saveModel(model_path)
    
    def update(self, data: DataLoader, clip: float, loss_fn = torch.nn.MSELoss()) -> float:
        losses = []
        for batch in tqdm(data):
            batch.to(self.device)
            pred = self.model(batch)
            if self.level == "sample":
                y = batch.y_sum
            else:
                y = batch.y_each
            loss = loss_fn(pred.squeeze(), y)
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
            if self.level == "sample":
                y = batch.y_sum
            else:
                y = batch.y_each
            loss = loss_fn(pred.squeeze(), y)
            losses.append(loss.item())
        return np.mean(losses)
    
    def saveModel(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    @staticmethod
    def savePerformance(losses: list[float], path: str) -> None:
        pd.DataFrame(losses).to_excel(path)

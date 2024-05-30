import numpy as np
import torch
from CrystalNet import LoadData
from torch_geometric.nn import DimeNet
from tqdm import tqdm
import pandas as pd

class Model(torch.nn.Module):
    def __init__(self, hidden_channels=16, 
                       num_blocks=2,
                       num_bilinear=2,
                       num_spherical=6,
                       num_radial=16,
                       cutoff=4.7):
        super().__init__()
        self.laye1 = DimeNet(hidden_channels, 1, num_blocks, num_bilinear, num_spherical, num_radial, cutoff=cutoff)

    def forward(self, data):
        out = self.laye1(data.z, data.pos)
        return torch.sum(out)

def train(model, data, optimizer, loss_fn, device):
    losses = []
    for batch in tqdm(data):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, torch.sum(batch.y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(np.sqrt(loss.item()))
    return np.mean(losses)

def validation(model, data, loss_fn, device):
    losses = []
    for batch in tqdm(data):
        batch.to(device)
        pred = model(batch)
        loss = loss_fn(pred, torch.sum(batch.y))
        losses.append(loss.item())
    return np.mean(losses)


# ========================================================================================================================================================================


data = "../../../../dataset/features/GNN/AtomicNumberPos_54"

fold_10_data = LoadData(data).KFold(fold=10)

hidden_channels = 64
num_bilinear = 5
num_blocks = 1
lr = 0.0001
epoch = 1000
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
device = torch.device("cuda:0")

for idx, (train_data, validation_data) in enumerate(fold_10_data):
    if idx < 5 or idx >=7:
        continue
    gnn_model = Model(hidden_channels=hidden_channels, num_blocks=num_blocks, num_bilinear=num_bilinear)
    gnn_model.to(device)
    optimizer=torch.optim.Adam(gnn_model.parameters(), lr=lr)
    losses = []
    for i in range(epoch):
        train_loss = train(gnn_model, train_data, optimizer, train_loss_fn, device)
        validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
        print(f"Epoch at {i}, training_loss is {train_loss}, test_loss is {validate_loss}")
        losses.append([train_loss, validate_loss])
    output_name = f"fold_{idx}.xlsx"
    pd.DataFrame(losses).to_excel(output_name)

    

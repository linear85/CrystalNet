import numpy as np
import torch
from CrystalNet import LoadData
from torch_geometric.nn import DimeNet
from tqdm import tqdm
import pandas as pd
import os

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
        return out

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


train_data, validation_data = LoadData(data).transformInput(partition=0.7)
hidden_channels_list = [16, 64]
num_bilinear_list = [1, 2, 5]
num_blocks_list = [1, 2, 5]
lr_list = [0.01, 0.001, 0.0001]
optimizer_list = [torch.optim.SGD, torch.optim.Adam]
epoch = 500
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
device = torch.device("cuda:0")
prev_list = os.listdir(os.getcwd())
prev_list = [i for i in prev_list if "xlsx" in i]
prev_list = []

hyper_list = []
for hidden_channels in hidden_channels_list:
    for num_blocks in num_blocks_list:
        for num_bilinear in num_bilinear_list:
            for lr in lr_list:
                for optimizer in optimizer_list:
                    hyper_list.append((hidden_channels, num_blocks, num_bilinear, lr, optimizer))
    
    
for (hidden_channels, num_blocks, num_bilinear, lr, optimizer) in hyper_list: #100-108; 75-80 laptop//  # 80-100 3090
    if optimizer == torch.optim.SGD:
        output_name = f"{hidden_channels}_{num_blocks}_{num_bilinear}_{lr}_SGD.xlsx"
    else:
        output_name = f"{hidden_channels}_{num_blocks}_{num_bilinear}_{lr}_Adam.xlsx"
    if output_name in prev_list:
        continue
    print(output_name)
    gnn_model = Model(hidden_channels=hidden_channels, num_blocks=num_blocks, num_bilinear=num_bilinear)
    gnn_model.to(device)
    cur_optimizer=optimizer(gnn_model.parameters(), lr=lr)
    losses = []
    for i in range(epoch):
        train_loss = train(gnn_model, train_data, cur_optimizer, train_loss_fn, device)
        validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
        print(f"Epoch at {i}, training_loss is {train_loss}, test_loss is {validate_loss}")
        losses.append([train_loss, validate_loss])
    pd.DataFrame(losses).to_excel(output_name)
    prev_list = os.listdir(os.getcwd())
    prev_list = [i for i in prev_list if "xlsx" in i]


import numpy as np
import torch
from CrystalNet import LoadData
from torch_geometric.nn import SchNet
from tqdm import tqdm
import pandas as pd
import os

class Model(torch.nn.Module):
    def __init__(self, hidden_channels=128, 
                       num_filters=128,
                       num_interactions=6,
                       num_gaussians=50,
                       cutoff=4.7):
        super().__init__()
        self.laye1 = SchNet(hidden_channels=hidden_channels, num_filters=num_filters, num_interactions=num_interactions, num_gaussians=num_gaussians, cutoff=cutoff)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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
epoch = 500
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
device = torch.device("cuda:0")

hidden_channels_list = [64, 128]
num_filters_list = [64, 128]
num_interactions_list = [3, 6]
num_gaussians_list = [20, 50]
lr_list = [0.01, 0.001, 0.0001]
optimizer_list = [torch.optim.SGD, torch.optim.Adam]
prev_list = os.listdir(os.getcwd())
prev_list = [i for i in prev_list if "xlsx" in i]

hyper_list = []
for hidden_channels in hidden_channels_list:
    for num_filters in num_filters_list:
        for num_interactions in num_interactions_list:
            for num_gaussians in num_gaussians_list:
                for lr in lr_list:
                    for optimizer in optimizer_list:
                        hyper_list.append((hidden_channels, num_filters, num_interactions, num_gaussians, lr, optimizer))

for (hidden_channels, num_filters, num_interactions, num_gaussians, lr, optimizer) in hyper_list:             
    if optimizer == torch.optim.SGD:
        output_name = f"{hidden_channels}_{num_filters}_{num_interactions}_{num_gaussians}_{lr}_SGD.xlsx"
    else:
        output_name = f"{hidden_channels}_{num_filters}_{num_interactions}_{num_gaussians}_{lr}_Adam.xlsx"
    if output_name in prev_list:
        continue
    print(output_name)
    gnn_model = Model(hidden_channels=hidden_channels, num_filters=num_filters, num_interactions=num_interactions, num_gaussians=num_gaussians)
    gnn_model.to(device)
    cur_optimizer=optimizer(gnn_model.parameters(), lr=lr)
    losses = []
    for i in range(epoch):
        train_loss = train(gnn_model, train_data, cur_optimizer, train_loss_fn, device)
        validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
        print(f"Epoch at {i}, training_loss is {train_loss}, test_loss is {validate_loss}")
        losses.append([train_loss, validate_loss])
    pd.DataFrame(losses).to_excel(output_name)

    

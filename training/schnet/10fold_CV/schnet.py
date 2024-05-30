import numpy as np
import torch
from CrystalNet import LoadData
from torch_geometric.nn import SchNet
from tqdm import tqdm
import pandas as pd

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
lr = 0.0001
epoch = 1000
hidden_channels = 64
num_filters = 64
num_interactions = 3
num_gaussians = 50
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
optimizer=torch.optim.Adam
device = torch.device("cuda:0")

for i, (train_data, validation_data) in enumerate(fold_10_data):
    if (i < 5):
        continue
    print("fold: ", i)
    gnn_model = Model(hidden_channels=hidden_channels, num_filters=num_filters, num_interactions=num_interactions, num_gaussians=num_gaussians)
    gnn_model.to(device)
    optimizer=torch.optim.SGD(gnn_model.parameters(), lr=lr)
    losses = []
    for idx in range(epoch):
        train_loss = train(gnn_model, train_data, optimizer, train_loss_fn, device)
        validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
        print(f"Epoch at {idx}, training_loss is {train_loss}, test_loss is {validate_loss}")
        losses.append([train_loss, validate_loss])
    output_name = f"fold_{i}.xlsx"
    pd.DataFrame(losses).to_excel(output_name)

    

import numpy as np
import torch
from CrystalNet import LoadData
from torch_geometric.nn import NNConv
from tqdm import tqdm
import pandas as pd


class EdgeModel(torch.nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size)
        )

    def forward(self, x) :
        return self.layer(x)


class Model(torch.nn.Module):
    def __init__(self, hidden_layer, edge_model_1, edge_mode_2):
        super().__init__()
        self.layer1 = NNConv(4, hidden_layer, edge_model_1)
        self.layer2 = NNConv(hidden_layer, 1, edge_mode_2)

    def forward(self, data):
        out = self.layer1(data.x, data.edge_index, data.edge_attr)
        out = torch.nn.ReLU()(out)
        out = self.layer2(out, data.edge_index, data.edge_attr)
        return out

def train(model, data, optimizer, loss_fn, device):
    losses = []
    for batch in tqdm(data):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
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
        loss = loss_fn(pred, batch.y)
        losses.append(loss.item())
    return np.mean(losses)


# ========================================================================================================================================================================


data_path = "../../../../dataset/features/GNN/OnehotDistance_54"

fold_10_data = LoadData(data_path).KFold(fold=10)
lr = 0.01
epoch = 1000
hidden_size = 512
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
optimizer=torch.optim.SGD
device = torch.device("cuda:0")

for i, (train_data, validation_data) in enumerate(fold_10_data):
    print("fold: ", i)
    if i != 4 and i != 6 and i != 8:
        continue
    edge_model_1 = EdgeModel(4*hidden_size)
    edge_model_2 = EdgeModel(hidden_size)
    gnn_model = Model(hidden_size, edge_model_1, edge_model_2)
    gnn_model.to(device)
    cur_optimizer=optimizer(gnn_model.parameters(), lr=lr)
    losses = []
    for idx in range(epoch):
        train_loss = train(gnn_model, train_data, cur_optimizer, train_loss_fn, device)
        validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
        print(f"Epoch at {idx}, training_loss is {train_loss}, test_loss is {validate_loss}")
        losses.append([train_loss, validate_loss])
    output_name = f"fold_{i}.xlsx"
    pd.DataFrame(losses).to_excel(output_name)

    

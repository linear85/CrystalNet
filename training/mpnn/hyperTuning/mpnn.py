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


data = "../../../../dataset/features/GNN/OnehotDistance_54"

print(torch.cuda.is_available())

train_data, validation_data = LoadData(data).transformInput(partition=0.7)
hidden_layer_list = [16, 64, 128, 512, 1024, 2048]
lr_list = [0.01, 0.001, 0.0001]
optimizer_list = [torch.optim.SGD]
epoch = 500
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
device = torch.device("cuda:0")

for hidden_layer in hidden_layer_list:
    for lr in lr_list:
        for optimizer in optimizer_list:
            if optimizer == torch.optim.SGD:
                output_name = f"{hidden_layer}_{lr}_SGD.xlsx"
            else:
                output_name = f"{hidden_layer}_{lr}_Adam.xlsx"
            edge_model_1 = EdgeModel(4*hidden_layer)
            edge_model_2 = EdgeModel(hidden_layer)
            gnn_model = Model(hidden_layer, edge_model_1, edge_model_2)
            gnn_model.to(device)
            cur_optimizer=optimizer(gnn_model.parameters(), lr=lr)
            losses = []
            for i in range(epoch):
                train_loss = train(gnn_model, train_data, cur_optimizer, train_loss_fn, device)
                validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
                print(f"Epoch at {i}, training_loss is {train_loss}, test_loss is {validate_loss}")
                losses.append([train_loss, validate_loss])
            pd.DataFrame(losses).to_excel(output_name)

    

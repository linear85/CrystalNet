import numpy as np
import torch
from CrystalNet import LoadData
from torch_geometric.nn import GraphConv
from tqdm import tqdm
import pandas as pd

class Model(torch.nn.Module):
    def __init__(self, hidden_layer):
        super().__init__()
        self.layer1 = GraphConv(4, hidden_layer)
        self.layer2 = GraphConv(hidden_layer, 1)

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


train_data_path = "../../../../dataset/features/GNN/OnehotDistance_54"
train_data, _ = LoadData(train_data_path).transformInput(partition=-1)

validation_data_path = "../../../../dataset/features/GNN/OnehotDistance_8192"
validation_data, _ = LoadData(validation_data_path).transformInput(partition=-1)

lr = 0.001
epoch = 1000
hidden_size = 64
train_loss_fn = torch.nn.MSELoss()
validation_loss_fn = torch.nn.L1Loss()
device = torch.device("cuda:0")


gnn_model = Model(hidden_size)
gnn_model.to(device)
optimizer=torch.optim.Adam(gnn_model.parameters(), lr=lr)
losses = []
for idx in range(epoch):
    train_loss = train(gnn_model, train_data, optimizer, train_loss_fn, device)
    validate_loss = validation(gnn_model, validation_data, validation_loss_fn, device)
    print(f"Epoch at {idx}, training_loss is {train_loss}, test_loss is {validate_loss}")
    losses.append([train_loss, validate_loss])
output_name = f"Losses.xlsx"
pd.DataFrame(losses).to_excel(output_name)

    

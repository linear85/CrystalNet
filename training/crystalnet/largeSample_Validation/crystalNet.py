from CrystalNet import trainer, ThreeBody, ThreeBody_Local, LoadData
import torch

train_data_path = "../../../../dataset/features/crystalNet/threeBody_54"
train_data, _ = LoadData(train_data_path).transformInput(partition=-1)

test_data_path = "../../../../dataset/features/crystalNet/threeBody_8192"
test_data, _ = LoadData(test_data_path).transformInput(partition=-1)


lr = 0.01
epoch = 5000
hidden_layer = 2048

model = ThreeBody_Local(hidden_size=hidden_layer)
agent = trainer(model, level="node", lr=lr)
cur_output = "Losses.xlsx"
agent.train(train_data, test_data, loss_path=cur_output, clip = 100, epoch=epoch)




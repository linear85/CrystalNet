from CrystalNet import trainer, ThreeBody, ThreeBody_Local, LoadData
import torch

train_data_path = "../../../../dataset/features/crystalNet/threeBody_54"
train_data, _ = LoadData(train_data_path).transformInput(partition=-1)

test_data_path = "../../../../dataset/features/crystalNet/threeBody_8192"
test_data, _ = LoadData(test_data_path).transformInput(partition=-1)

model_path = "CrystalNet_Model_2"

lr = 0.01
epoch = 5000
hidden_layer = 2048
criterion = [0.008, 0.008]

model = ThreeBody_Local(hidden_size=hidden_layer)
agent = trainer(model, level="node", lr=lr)
cur_output = "Losses_epoch_5000.xlsx"
agent.train(train_data, test_data, loss_path=cur_output, clip = 100, model_path=model_path, epoch=epoch, save_model_criteria=criterion)




import time
import pandas as pd
from CrystalNet import ThreeBody_Local, trainer, ThreeBody, LoadData
import torch


data = "../../../../dataset/features/crystalNet/threeBody_54"
train_data, validation_data = LoadData(data).transformInput(partition=0.7)
start_time = time.perf_counter()

epoch = 500
hidden_layer_list = [16, 64, 128, 512, 1024, 2048]
lr_list = [0.01, 0.001, 0.0001]
optimizer_list = [torch.optim.SGD, torch.optim.Adam]

for hidden_layer in hidden_layer_list:
    for lr in lr_list:
        for optimizer in optimizer_list:
            model = ThreeBody_Local(hidden_size=hidden_layer)
            agent = trainer(model, level="node", optimizer=optimizer, lr=lr)
            if optimizer == torch.optim.SGD:
                cur_output = f"{hidden_layer}_{lr}_SGD.xlsx"
            else:
                cur_output = f"{hidden_layer}_{lr}_Adam.xlsx"
            print(cur_output)
            agent.train(train_data, validation_data, loss_path=cur_output, epoch=epoch)
   

end_time = time.perf_counter()
print((end_time-start_time)//60)

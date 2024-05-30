import time
import pandas as pd
from CrystalNet import ThreeBody_Local, trainer, ThreeBody, LoadData



data_path = "../../../../dataset/features/crystalNet/threeBody_54"
fold_10_data = LoadData(data_path).KFold(fold=10)
start_time = time.perf_counter()

output_name = "threeBody_10FCV"
lr = 0.01
epoch = 1000
hidden_layer = 2048

for i, (train_data, test_data) in enumerate(fold_10_data):
    print("fold: ", i)
    model = ThreeBody_Local(input_size=12, hidden_size=hidden_layer)
    agent = trainer(model, level="node", lr=lr)
    cur_output = f"{output_name}_fold_{i}.xlsx"
    agent.train(train_data, test_data, loss_path=cur_output, epoch=epoch)
   

end_time = time.perf_counter()
print((end_time-start_time)//60)

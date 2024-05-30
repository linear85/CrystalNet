import pandas as pd 
import numpy as np
import os

def getBestPerformance(input_path: str) -> list[list[float]]:
    cur_df = pd.read_excel(input_path)
    data = cur_df.iloc[-50:,:]
    return data.mean().tolist()[1:] + data.std().tolist()[1:]

def getParameter(name: str) -> list[str]:
    pos_1 = name.index("_")
    v1 = float(name[:pos_1])
    name = name[pos_1+1:]
    pos_2 = name.index("_")
    v2 = float(name[:pos_2])
    name = name[pos_2+1:]
    pos_3 = name.index(".")
    v3 = name[:pos_3]
    return [v1, v2, v3]

cur_dir = os.getcwd()
data_list = os.listdir(cur_dir)
data_list = [i for i in data_list if "xlsx" in i and "_" in i and "0" in i]
res = []
for data in data_list:
    parameter = getParameter(data)
    performance = getBestPerformance(data)
    # print(parameter, performance)
    res.append(parameter + performance)

df = pd.DataFrame(res)
df.columns=["hidden_layer_size", "lr", "optimizer", "training_loss_mean", "validation_loss_mean", "training_loss_std", "validation_loss_std"]
df.to_excel("HyperparameterTuning_average_performance.xlsx")
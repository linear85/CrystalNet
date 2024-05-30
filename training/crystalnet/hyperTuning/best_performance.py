import pandas as pd 
import numpy as np
import os

def getBestPerformance(input_path: str) -> list[list[float]]:
    cur_df = pd.read_excel(input_path)
    idx = cur_df.iloc[:, 2].idxmin()
    cur_res = cur_df.iloc[idx].tolist()
    return cur_res[1:]

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
data_list = [i for i in data_list if "xlsx" in i]
res = []
for data in data_list:
    parameter = getParameter(data)
    performance = getBestPerformance(data)
    res.append(parameter + performance)

df = pd.DataFrame(res)
df.columns=["hidden_layer_size", "lr", "optimizer", "training_loss", "validation_loss"]
df.to_excel("HyperparameterTuning.xlsx")
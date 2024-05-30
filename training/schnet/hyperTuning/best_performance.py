import pandas as pd 
import numpy as np
import os

def getBestPerformance(input_path: str) -> list[float]:
    cur_df = pd.read_excel(input_path)
    idx = cur_df.iloc[:, 2].idxmin()
    cur_res = cur_df.iloc[idx].tolist()
    res = cur_res[1:]
    return [i/54 for i in res]

def getParameter(name: str) -> list[str]:
    pos_1 = name.index("_")
    v1 = float(name[:pos_1])
    name = name[pos_1+1:]
    pos_2 = name.index("_")
    v2 = float(name[:pos_2])
    name = name[pos_2+1:]
    pos_3 = name.index("_")
    v3 = float(name[:pos_3])
    name = name[pos_3+1:]
    pos_4 = name.index("_")
    v4 = float(name[:pos_4])
    name = name[pos_4+1:]
    pos_5 = name.index("_")
    v5 = float(name[:pos_5])
    name = name[pos_5+1:]
    pos_6 = name.index(".")
    v6 = name[:pos_6]
    return [v1, v2, v3, v4, v5, v6]

cur_dir = os.getcwd()
data_list = os.listdir(cur_dir)
data_list = [i for i in data_list if "xlsx" in i]
res = []
for data in data_list:
    print(data)
    parameter = getParameter(data)
    performance = getBestPerformance(data)
    res.append(parameter + performance)


df = pd.DataFrame(res)
df.columns=["hidden_channels", "num_filters",  "num_interactions", "num_gaussians", "lr", "optimizer", "training_loss", "validation_loss"]
df.to_excel("HyperparameterTuning.xlsx")
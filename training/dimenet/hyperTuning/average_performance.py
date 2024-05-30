import pandas as pd 
import numpy as np
import os

def getBestPerformance(input_path: str) -> list[list[float]]:
    cur_df = pd.read_excel(input_path)
    data = cur_df.iloc[-50:,:]/54
    return data.mean().tolist()[1:] + data.std().tolist()[1:]

def getParameter(name: str) -> list[str]:
    res = name.split("_")
    return res[:-1] + [res[-1][:-5]]

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
df.columns=["idden_channels", "bilinear", "blocks", 'lr', "optimizer", "training_loss_mean", "validation_loss_mean", "training_loss_std", "validation_loss_std"]
df.to_excel("HyperparameterTuning_average_performance.xlsx")
# print(df)
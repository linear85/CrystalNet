import pandas as pd 
import numpy as np
import os

def getBestPerformance(input_path: str) -> list[list[float]]:
    cur_df = pd.read_excel(input_path)
    data = cur_df.iloc[-50:,:]
    return data.mean().tolist()[1:] + data.std().tolist()[1:]

def get_10fold_CV(cur_dir: str):
    data_list = os.listdir(cur_dir)
    data_list = [i for i in data_list if "xlsx" in i]
    res = []
    for data in data_list:
        cur_data = os.path.join(cur_dir, data)
        performance = getBestPerformance(cur_data)
        res.append(performance)
    data = np.array(res)
    if "dimenet" in cur_dir or "schnet" in cur_dir:
        data = data/54
    return list(np.mean(data, axis=0))[:2] + list(np.std(data, axis=0))[:2]

cur_path = os.getcwd()
model_list = os.listdir(cur_path)
model_list = [i for i in model_list if "." not in i]
output = []

for model in model_list:
    path_10CV = os.path.join(cur_path, model, "10fold_CV")
    cur_res = get_10fold_CV(path_10CV)
    output.append([model] + cur_res)

df = pd.DataFrame(output)
df.columns = ["models", "training_MAE_mean", "test_MAE_mean", "training_std", "test_std"]
df.to_excel("10Fold_CV.xlsx")
import pandas as pd 
import numpy as np
import os

def getBestPerformance(input_path: str) -> list[list[float]]:
    cur_df = pd.read_excel(input_path)
    data = cur_df.iloc[-50:,:]
    if "dimenet" in input_path or "schnet" in input_path:
        data = data/54
    return data.mean().tolist()[1:] + data.std().tolist()[1:]



cur_path = os.getcwd()
model_list = os.listdir(cur_path)
model_list = [i for i in model_list if "." not in i]
output = []

for model in model_list:
    path_10CV = os.path.join(cur_path, model, "largeSample_Validation", "Losses.xlsx")
    cur_res = getBestPerformance(path_10CV)
    output.append([model] + cur_res)

df = pd.DataFrame(output)
df.columns = ["models", "training_MAE_mean", "test_MAE_mean", "training_std", "test_std"]
print(df)
df.to_excel("largeSampleValidation.xlsx")
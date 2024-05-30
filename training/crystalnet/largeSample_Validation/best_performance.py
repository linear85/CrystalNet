import pandas as pd 
import numpy as np

def getBestPerformance(input_path: str) -> list[list[float]]:
    xl = pd.ExcelFile(input_path)
    sheets = xl.sheet_names
    res = []
    for sheet in sheets:
        cur_df = xl.parse(sheet)
        idx = cur_df.iloc[:, 2].idxmin()
        cur_res = cur_df.iloc[idx].tolist()
        res.append(cur_res[1:])
    print(np.mean(res, axis=0))
    print(np.std(res, axis=0))

data_path = "Losses.xlsx"
getBestPerformance(data_path)
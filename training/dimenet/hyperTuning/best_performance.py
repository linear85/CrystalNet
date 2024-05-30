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
    return list(np.mean(res, axis=0))


hidden_layer_list = [16, 64, 128, 512, 1024, 2048]
hidden_layer_list = [16, 64, 128, 512]
num_blocks_list = [1, 2, 5, 10]
res = []
for blocks in num_blocks_list:
    data_path = f"{blocks}.xlsx"
    data_path = f"num_blocks_{blocks}.xlsx"
    res.append(getBestPerformance(data_path))

pd.DataFrame(res).to_excel("num_blocks.xlsx")
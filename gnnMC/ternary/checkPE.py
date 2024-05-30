from CrystalNet import ThreeBody_Local, ThreeBodyFeature
import torch
import pandas as pd
import numpy as np
import os

def getPredict(model, data, PE=None, saveName=False)->float:
    if PE:
        data_gnn, _, _, _ = ThreeBodyFeature(data, PE=PE).threeBodyFeatures()
        prediction = model(data_gnn)
        result = torch.sum(prediction.squeeze())
        if saveName:
            np.save(saveName, prediction.squeeze().detach().numpy())
        return result.item()
    else:
        data_gnn, _, _, _ = ThreeBodyFeature(data).threeBodyFeatures()
        if saveName:
            np.save(saveName, data_gnn.y_each)
        return data_gnn.y_sum.item()

def gnnMC_Analysis(folder: str, model_path: str, alloy: str) -> None:
    gnn_dump_files = ["dump_0", "dump_1000000"]
    lammps_dump_files = ["lammps_dump_0", "lammps_dump_1000000"]
    model = ThreeBody_Local(hidden_size=2048)
    model.load_state_dict(torch.load(model_path))
    res = []
    for idx in range(len(gnn_dump_files)):
        print(gnn_dump_files[idx], lammps_dump_files[idx])
        gnn_data = os.path.join(folder, gnn_dump_files[idx])
        lammps_data = os.path.join(folder, lammps_dump_files[idx])
        if idx == 10:
            gnn_pe = getPredict(model, gnn_data, PE=[-1], saveName=f"{alloy}_gnn.npy")
            lammps_pe = getPredict(model, lammps_data, saveName=f"{alloy}_lammps.npy")
        else:
            gnn_pe = getPredict(model, gnn_data, PE=[-1])
            lammps_pe = getPredict(model, lammps_data)
        res.append([gnn_pe, lammps_pe, abs(gnn_pe-lammps_pe)/8192])
    return res


cur_folder = os.getcwd()
alloy_list = os.listdir(cur_folder)
alloy_list = [i for i in alloy_list if "0" in i and "_" in i and "." not in i]
# print(alloy_list)
writer = pd.ExcelWriter("ternary_performance.xlsx")
model_path = r"C:\Users\89721\Dropbox (ASU)\projects\CrystalNet\train\node\crystalnet\model\CrystalNet_Model_1"

for alloy in alloy_list:
    print(alloy)
    dump_folder = os.path.join(cur_folder, alloy)
    res = gnnMC_Analysis(dump_folder, model_path, alloy)
    cur_df = pd.DataFrame(res)
    cur_df.columns= ["gnn", 'lammps', 'Diff']
    cur_df.to_excel(writer, sheet_name=alloy)

writer.close()


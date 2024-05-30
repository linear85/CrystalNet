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
    gnn_dump_files = [f"dump_{str(i)}" for i in range(0, 1000001, 100000)]
    lammps_dump_files = [f"lammps_dump_{str(i)}" for i in range(0, 1000001, 100000)]
    model = ThreeBody_Local(hidden_size=2048)
    model.load_state_dict(torch.load(model_path))
    res = []
    for idx in range(len(gnn_dump_files)):
        print(idx)
        print(gnn_dump_files[idx], lammps_dump_files[idx])
        gnn_data = os.path.join(folder, gnn_dump_files[idx])
        lammps_data = os.path.join(folder, lammps_dump_files[idx])
        if idx == 10:
            gnn_pe = getPredict(model, gnn_data, PE=[-1], saveName=f"{alloy}_gnn.npy")
            lammps_pe = getPredict(model, lammps_data, saveName=f"{alloy}_lammps.npy")
        else:
            gnn_pe = getPredict(model, gnn_data, PE=[-1])
            lammps_pe = getPredict(model, lammps_data)
        res.append([gnn_pe, lammps_pe])
        
    return res

# alloy = "25_10_25_40"
# alloy = "25_25_25_25"
alloy = "duplicate_Nb40"

dump_folder = os.path.join(os.getcwd(), alloy)


model_path = r"C:\Users\89721\Dropbox (ASU)\projects\CrystalNet\train\node\crystalnet\model\CrystalNet_Model_1"
res = gnnMC_Analysis(dump_folder, model_path, alloy)
pd.DataFrame(res).to_excel(f"{alloy}.xlsx")




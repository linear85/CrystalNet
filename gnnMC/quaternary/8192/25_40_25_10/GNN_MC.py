from CrystalNet import GNN_MC

dump_path = "25_40_25_10.lmp"
model_path = "../../train/node/crystalnet/model/CrystalNet_Model_1"
steps = 1000000
model = GNN_MC(dump_path, model_path)
model.run(steps, dump_step=10000)
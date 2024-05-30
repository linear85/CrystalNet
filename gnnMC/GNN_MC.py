from CrystalNet import GNN_MC

dump_path = "25_25_25_25.lmp"
model_path = "../train/crystalnet/model/CrystalNet_Model_1"
output_path = "1K.dump"
steps = 1000000
model = GNN_MC(dump_path, model_path)
model.run(steps, dump_step=10000)
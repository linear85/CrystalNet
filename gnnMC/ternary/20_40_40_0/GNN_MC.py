from CrystalNet import GNN_MC

dump_path = "input_structure.lmp"
model_path = "/anvil/scratch/x-yao1/CrystalNet/train/node/crystalnet/model/CrystalNet_Model_1"
# model_path = r"C:\Users\89721\Dropbox (ASU)\projects\CrystalNet\train\node\crystalnet\model\CrystalNet_Model_1"
steps = 1000000
model = GNN_MC(dump_path, model_path)
model.run(steps, dump_step=10000)
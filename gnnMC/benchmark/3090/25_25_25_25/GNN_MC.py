from CrystalNet import GNN_MC
import time

start = time.perf_counter()

dump_path = "25_25_25_25.lmp"
model_path = r"C:\Users\89721\Dropbox (ASU)\projects\CrystalNet\train\node\crystalnet\model\CrystalNet_Model_1"
steps = 1000
model = GNN_MC(dump_path, model_path)
model.run(steps, dump_step=10000)

end = time.perf_counter()
print((end-start))
from lammps import lammps
import os
import time

def runLammps(structure_path: str, output_path: str) -> None:
    with open("base.in", 'r') as f:
        input_script = f.read()
        new_input_script = input_script.format(structure_path)
    with open("tmp.in", 'w') as f:
        f.write(new_input_script)

    lmp = lammps()
    input_file = "tmp.in"
    lmp.file(input_file)
    # time.sleep(5)
    os.rename("mc.0.dump", output_path)
    os.remove("tmp.in")
    # os.remove("log.lammps")
    # time.sleep(5)

structure_dir = "../SROTuner_Nb40"
structure_files = os.listdir(structure_dir)
structure_files = [i for i in structure_files if "dump" in i and "0" in i]

for structure in structure_files:
    structure_path = os.path.join(structure_dir, structure)
    output_path = os.path.join(structure_dir, f"lammps_{structure}")
    runLammps(structure_path, output_path)
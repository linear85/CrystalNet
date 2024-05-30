from ovito.io import import_file 

def getPE(input_path: str) -> float:
    pipeline = import_file(input_path)
    data = pipeline.compute()
    return sum(data.particles['c_eng'][...])/data.particles.count

# input_path = "duplicate_Nb40/lammps_dump_1000000.lmp"
# input_path = "SROTuner_Nb40/lammps_dump_1000000.lmp"
input_path = "25_40_25_10/lammps_dump_1000000"
pe = getPE(input_path)
print(pe)
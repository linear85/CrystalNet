import os

dir_path = os.path.dirname(os.path.realpath(__file__))

working_files = os.listdir(dir_path)
# working_files.remove("10_20_70_0")
working_files = [i for i in working_files if "0" in i and "_" in i and "." not in i]


for cur_file in working_files:
    forceDir = os.path.join(dir_path, cur_file)
    os.chdir(forceDir)
    os.system("pwd")
    cmd = 'sbatch anvilPython.sh'
    os.system(cmd)

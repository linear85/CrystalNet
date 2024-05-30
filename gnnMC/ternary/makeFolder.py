import os
import shutil

def getCompositions():
    res = []
    for x1 in range(10, 80, 10):
        for x2 in range(10, 80, 10):
            for x3 in range(10, 80, 10):
                if (x1 + x2 + x3 == 100):
                    res.append([x1, x2, x3, 0])
    return res

compositions = getCompositions()
cur_path = os.getcwd()
for com in compositions:
    name = [str(i) for i in com]
    name = "_".join(name)
    # src = os.path.join(cur_path, "initial_structure", f"{name}.lmp")
    # print(src)
    # dst = os.path.join(cur_path, name, "input_structure.lmp")
    src = 'anvilPython.sh'
    dst =  os.path.join(cur_path, name, "anvilPython.sh")
    shutil.copy(src, dst)
    


from CrystalTool import changeComp

def getCompositions(step = 10):
    res = []
    for x1 in range(10, 80, 10):
        for x2 in range(10, 80, 10):
            for x3 in range(10, 80, 10):
                if (x1 + x2 + x3 == 100):
                    res.append([x1, x2, x3, 0])
    return res

step = 5
composition_list = getCompositions(step)
# composition_list = [[25, 25, 25, 25], [25, 10, 25, 40], [25, 40, 25, 10]]
input_file = "100_0_0_0.lmp"
for comp in composition_list:
    name = "_".join([str(i) for i in comp])
    output_dir = f"{name}.lmp"
    changeComp(input_file, output_dir, comp)


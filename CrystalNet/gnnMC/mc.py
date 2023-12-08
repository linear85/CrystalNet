import random
from tqdm import tqdm
import os
from CrystalNet.feature import ThreeBodyFeature
from CrystalNet.model import ThreeBody
from ovito.io import import_file
import torch
from torch_geometric.loader import DataLoader
import time
import pandas as pd


class node:
    def __init__(self, _type: int, cff_index: list[tuple[int]] = [], cfs_index: list[tuple[int]] = []) -> None:
        self._type = _type
        self.cff_index = cff_index
        self.cfs_index = cfs_index


class GNN_MC:
    def __init__(self, dump_path: str, model_path: str) -> None:
        pipeline = import_file(dump_path)
        self.ovito_data = pipeline.compute()
        self._types = list(self.ovito_data.particles.particle_types)
        self.feature, cff_index, cfs_index = Feature(dump_path).threeBodyFeatures()
        print("read initial structure done!")
        self.nodes = self.__mapNode(cff_index, cfs_index)
        print("map node done!")
        self.model = self.readModel(model_path)
        self.cur_PE = self.model_prediction()
    
    def __mapNode(self, cff_index, cfs_index):
        nodes = [node(i, [], []) for i in self._types]
        for idx in range(len(cff_index)):
            cur_index = self.__convertIndex(idx)
            node_index = cff_index[idx]
            cur_node = nodes[node_index]
            cur_node.cff_index.append(cur_index)
        for idx in range(len(cfs_index)):
            cur_index = self.__convertIndex(idx)
            node_index = cfs_index[idx]
            cur_node = nodes[node_index]
            cur_node.cfs_index.append(cur_index)
       
        return nodes

    @staticmethod
    def __convertIndex(idx: int) -> tuple[int]:
        n1 = idx        //  (3 * 24)
        n2 = (idx // 3) %   24
        n3 = idx        %   3
        return (n1, n2, n3)

    def run(self, steps: int, out_step: int = 100) -> None:
        for idx in tqdm(range(steps)):
            atom_1, atom_2 = self.randomTwoAtoms()
            self.swap(atom_1, atom_2)
            self.accept(atom_1, atom_2)
            if idx % out_step == 0:
                print(f"{idx} step: {self.cur_PE}")

    @staticmethod
    def readModel(path: str) -> torch.nn:
        model = ThreeBody()
        model.load_state_dict(torch.load(path))
        return model

    def model_prediction(self) -> torch.tensor:
        cur_node = self.feature.clone()
        cur_node.x_cff = cur_node.x_cff.reshape((8192, 24, 12))
        cur_node.x_cfs = cur_node.x_cfs.reshape((8192, 24, 12))
        return self.model(cur_node)
            
    def accept(self, atom_1: int, atom_2: int) -> None:
        next_PE = self.model_prediction()
        if (next_PE < self.cur_PE):
            self.cur_PE = next_PE
        else:
            self.swap(atom_1, atom_2)
    
    def swap(self, atom_1_pos: int, atom_2_pos: int) -> None:
        atom_1_type = self.nodes[atom_1_pos]._type
        atom_2_type = self.nodes[atom_2_pos]._type
        self.__changeAtom(atom_1_pos, atom_2_type)
        self.__changeAtom(atom_2_pos, atom_1_type)
  

    def __changeAtom(self, pos: int, _type: int) -> None:
        self.nodes[pos]._type = _type
        onehot_type = self.toOneHot(_type)
        for idx in self.nodes[pos].cff_index:
            self.feature.x_cff[idx] = onehot_type
        for idx in self.nodes[pos].cfs_index:
            self.feature.x_cfs[idx] = onehot_type
    
    def randomTwoAtoms(self) -> None:
        while True:
            atom_1 = random.randint(0, self.ovito_data.particles.count-1)
            atom_2 = random.randint(0, self.ovito_data.particles.count-1)
            if (atom_2 != atom_1 and self.nodes[atom_1]._type != self.nodes[atom_2]._type):
                return atom_1, atom_2

    def changeComp(self, template: str, output_path: str, types: list[int]) -> None:
        s = open(template, 'r')
        First_Part = ""
        while True:
            line = s.readline()
            if ("Masses" in line):
                for _ in range(5):
                    s.readline()
                continue
            else:
                First_Part = First_Part + line
            if ("Atoms # atomic" in line):
                First_Part = First_Part + '\n'
                break
        df = pd.read_csv(s, header=None, delimiter=' +', engine='python')
        s.close()
        df.iloc[:, 1] = types
        df.to_csv("tmp.lmp", header=None, index=None, sep=' ', mode='a')
        f_tmp = open("tmp.lmp", 'r')
        text_Second_part = f_tmp.read()
        f_tmp.close()
        f_write = open(output_path, 'w')
        text = First_Part + text_Second_part
        f_write.write(text)
        f_write.close()
        os.remove("tmp.lmp")

    def saveStructure(self, output_path: str, output_template = "template.data") -> None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        output_template = os.path.join(base_path, output_template)
        types = [i._type for i in self.nodes]
        self.changeComp(output_template, output_path, types)

    @staticmethod
    def toOneHot(atom_type: int) -> torch.tensor:
        feature = [0, 0, 0, 0]
        feature[atom_type - 1] = 1
        return torch.tensor(feature, dtype=torch.float)
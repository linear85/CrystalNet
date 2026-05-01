from torch_geometric.data.data import Data
from ovito.io import import_file
from ovito.data import NearestNeighborFinder, CutoffNeighborFinder
import torch
from tqdm import tqdm


class ThreeBodyFeature:
    def __init__(self, file: str, structure: str="BCC", PE=None) -> None:
        pipeline = import_file(file)
        self.data = pipeline.compute()
        if not PE:
            self.PE = self.data.particles['c_eng'][...]
        else:
            self.PE = PE
        if structure == "BCC":
            num_neigh = 26
        elif structure == "FCC":
            num_neigh = 42
        else:
            raise Exception("Only support BCC or FCC crystal structure, but get ", structure)
        self.structure = structure
        self.finder_third = NearestNeighborFinder(num_neigh, self.data)
        self.types = list(self.data.particles.particle_types)
        self.graph = self.toGraph()
    
    
    def toGraph(self) -> list[list[set[int]]]:
        graph = []
        for idx in range(self.data.particles.count):
            neighbor = list(self.finder_third.find(idx))
            neighbor = [i.index for i in neighbor]
            if self.structure == "BCC":
                sub_graph = [set(neighbor[:8]),  set(neighbor[8:14]),  set(neighbor[14:])]
            elif self.structure == "FCC":
                sub_graph = [set(neighbor[:12]), set(neighbor[12:18]), set(neighbor[18:])]
            graph.append(sub_graph)
        return graph
    
    
    def threeBodyFeatures(self) -> tuple[Data, list[int], list[int]]:
        cff_1       = []
        cff_2       = []
        cfs         = []
        cff_1_index = []
        cff_2_index = []
        cfs_index   = []
        for idx in range(self.data.particles.count):
            [first, second, _] = self.graph[idx]
            cur_cff_1, cur_cff_2, cur_cff_1_index, cur_cff_2_index = self.__get_cff_1_2(idx, first)
            cur_cfs, cur_cfs_index = self.__get_cfs(idx, first, second)
            cff_1.append(cur_cff_1)
            cff_2.append(cur_cff_2)
            cfs.append(cur_cfs)
            cff_1_index.extend(cur_cff_1_index)
            cff_2_index.extend(cur_cff_2_index)
            cfs_index.extend(cur_cfs_index)
        print(len(cff_1))
        x_cff_1 = torch.tensor(cff_1, dtype=torch.float)
        x_cff_2 = torch.tensor(cff_2, dtype=torch.float)
        x_cfs = torch.tensor(cfs, dtype=torch.float)
        y_sum =  torch.tensor([float(sum(self.PE))], dtype=torch.float)
        y_each = torch.tensor(list(self.PE), dtype=torch.float)
        cur_gnn = Data(x_cff_1=x_cff_1, x_cff_2=x_cff_2, x_cfs=x_cfs, y_sum=y_sum, y_each=y_each)
        return cur_gnn, cff_1_index, cff_2_index, cfs_index
    

    def __get_cff_1_2(self, center_idx: int, first: set[int])-> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
        cur_cff_1 = []
        cur_cff_1_index = []
        cur_cff_2 = []
        cur_cff_2_index = []
        for neighbor in first:
            tmp_cff_1, tmp_index_1 = self.__cff(center_idx, neighbor, self.graph[neighbor][1], first)
            tmp_cff_2, tmp_index_2 = self.__cff(center_idx, neighbor, self.graph[neighbor][2], first)
            cur_cff_1.extend(tmp_cff_1)
            cur_cff_2.extend(tmp_cff_2)
            cur_cff_1_index.extend(tmp_index_1)
            cur_cff_2_index.extend(tmp_index_2)
        return cur_cff_1, cur_cff_2, cur_cff_1_index, cur_cff_2_index
    

    def __get_cfs(self, center_idx: int, first: set[int], second: set[int]) -> tuple[list[list[int]], list[int]]:
        cur_cfs = []
        cfs_index = []
        for neighbor in first:
            tmp_cfs, index_list = self.__cff(center_idx, neighbor, self.graph[neighbor][0], second)
            cur_cfs.extend(tmp_cfs)
            cfs_index.extend(index_list)
        return cur_cfs, cfs_index
    
    
    def __cff(self, center_idx: int, neighbor_idx: int, neighbor_list: set[int], prev: set[int]) -> tuple[list[list[int]], list[int]]:
        cur_cff = []
        index_list = []
        for nei in neighbor_list:
            if nei in prev:
                tmp = self.toOneHot(self.types[center_idx]) + self.toOneHot(self.types[neighbor_idx]) + self.toOneHot(self.types[nei])
                cur_cff.append(tmp)
                index_list.extend([center_idx, neighbor_idx, nei])
        return cur_cff, index_list


    @staticmethod
    def toOneHot(atom_type: int) -> list[int]:
        feature = [0, 0, 0, 0]
        feature[atom_type - 1] = 1
        return feature
    

  

class Dump2Data:
    def __init__(self, file: str, cutoff: float=4.7, PE=None) -> None:
        pipeline = import_file(file)
        self.data = pipeline.compute()
        if not PE:
            self.PE = self.data.particles['c_peratom'][...]
        else:
            self.PE = PE
        self.finder         = CutoffNeighborFinder(cutoff, self.data)
        self.types          = list(self.data.particles.particle_types)
        self.edge_index     = []
        self.edge_weights   = []
    
    def to_data(self) -> Data:
        edge_index     = [[], []]
        edge_weights   = []
        atomic_numbers = []
        for idx in tqdm(range(self.data.particles.count)):
            cur_type = self.types[idx]
            cur_z    = self.atom_type_2_atom_number(cur_type)
            atomic_numbers.append(cur_z)
            for neigh in self.finder.find(idx):
                edge_index[0].append(neigh.index)
                edge_index[1].append(idx)
                edge_weights.append(neigh.distance)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        y_sum =  torch.tensor([float(sum(self.PE))], dtype=torch.float)
        y_each = torch.tensor(list(self.PE), dtype=torch.float)
        cur_gnn = Data(edge_index=edge_index, edge_weights=edge_weights, atomic_numbers=atomic_numbers, y_sum=y_sum, y_each=y_each)
        return cur_gnn

    @staticmethod
    def atom_type_2_atom_number(atom_type: int) -> int:
        atomic_number = None
        if atom_type == 1:
            atomic_number = 42
        elif atom_type == 2:
            atomic_number = 41
        elif atom_type == 3:
            atomic_number = 73
        elif atom_type == 4:
            atomic_number = 74
        elif atom_type == 5:
            atomic_number = 23
        return atomic_number




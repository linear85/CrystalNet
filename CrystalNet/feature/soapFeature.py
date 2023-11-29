from torch_geometric.data.data import Data
from ovito.io import import_file
from ovito.data import NearestNeighborFinder
import torch
from dscribe.descriptors import SOAP
from ovito.io.ase import ovito_to_ase

class SOAPFature:
    def __init__(self, file: str, PE=None):
        pipeline = import_file(file)
        self.data = pipeline.compute()
        if not PE:
            self.PE = self.data.particles['c_eng'][...]
        else:
            self.PE = PE
    
    def getSOAP(self, n_max: int, l_max: int) -> list[Data]:
        ase_data = ovito_to_ase(self.data)
        species = ["Nb", "Mo", "Ta", "W"]
        cut_off = 4.7
        soap_desc = SOAP(species=species, r_cut=cut_off, n_max=n_max, l_max=l_max, periodic=True)
        soap = soap_desc.create(ase_data)
        features = []
        for i in range(len(soap)):
            cur_x = torch.tensor(soap[i], dtype=torch.float)
            cur_y = torch.tensor(self.PE[i], dtype=torch.float)
            cur_gnn = Data(x=cur_x, y=cur_y, num_nodes=1)
            features.append(cur_gnn)
        return features



  




import torch


class SOAP(torch.nn.Module):
    def __init__(self, input_dim: str) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024*6),
            torch.nn.ReLU(),
            torch.nn.Linear(1024*6, 1024*3),
            torch.nn.ReLU(),
            torch.nn.Linear(1024*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layers(x)

class ThreeBody(torch.nn.Module):
    def __init__(self, input_size=12, hidden_size=500) -> None:
        super().__init__()
        self.layers_cff = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.layers_cfs = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, batch) -> torch.tensor:
        out_cff = torch.sum(self.layers_cff(batch.x_cff_1), axis=1)
        out_cfs   = torch.sum(self.layers_cfs(batch.x_cfs), axis=1)
        return torch.sum(self.out(torch.concat((out_cff, out_cfs), axis=1)))


class ThreeBody_cff_1_2_cfs(torch.nn.Module):
    def __init__(self, input_size=12, hidden_size=500) -> None:
        super().__init__()
        self.layers_cff_1 = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.layers_cff_2 = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.layers_cfs   = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*3, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, batch) -> torch.tensor:
        out_cff_1 = torch.sum(self.layers_cff_1(batch.x_cff_1), axis=1)
        out_cff_2 = torch.sum(self.layers_cff_2(batch.x_cff_1), axis=1)
        out_cfs   = torch.sum(self.layers_cfs(batch.x_cfs), axis=1)
        return torch.sum(self.out(torch.concat((out_cff_1, out_cff_2, out_cfs), axis=1)))


class TreeBody_Local(ThreeBody):
    def forward(self, batch) -> torch.tensor:
        out_cff = torch.sum(self.layers_cff(batch.x_cff_1), axis=1)
        out_cfs   = torch.sum(self.layers_cfs(batch.x_cfs), axis=1)
        return self.out(torch.concat((out_cff, out_cfs), axis=1))


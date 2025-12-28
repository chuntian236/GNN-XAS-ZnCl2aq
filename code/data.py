import torch
from torch.utils.data import Dataset
import dgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.compute import compute_pair_vector_and_distance

class XASGraphDataset(Dataset):
    """
    Dataset for XASLightningModule:
    - Converts pymatgen structures into DGL graphs
    - Stores state features (optional)
    - Stores target spectra
    """

    def __init__(self, structures, spectra, element_types=("O", "H", "Zn", "Cl"), cutoff=4.0):
        """
        Args:
            structures: list of pymatgen Structure objects
            spectra: torch.Tensor or np.ndarray of shape (N, spectrum_len)
            element_types: tuple of allowed elements
            cutoff: cutoff radius for neighbor graph
        """
        self.converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
        self.graphs, self.states, self.spectra = [], [], []

        for s, spec in zip(structures, spectra):
            g, lat, state = self.converter.get_graph(s)

            g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
            g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]

            bond_vec, bond_dist = compute_pair_vector_and_distance(g)
            g.edata["bond_vec"] = bond_vec
            g.edata["bond_dist"] = bond_dist

            self.graphs.append(g)
            self.states.append(torch.tensor(state, dtype=torch.float32))
            self.spectra.append(torch.tensor(spec, dtype=torch.float32))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.states[idx], self.spectra[idx]


class PrecomputedGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, states, spectra):
        self.graphs = graphs
        self.states = states
        self.spectra = spectra

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.states[idx], self.spectra[idx]


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
        

def collate_fn(batch):
    graphs, states, spectra = zip(*batch)
    bg = dgl.batch(graphs)
    states = torch.stack(states)
    spectra = torch.stack(spectra)
    return bg, states, spectra



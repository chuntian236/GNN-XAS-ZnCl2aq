import torch
import torch.nn as nn

import lightning.pytorch as pl

import itertools

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.layers import (
    MLP as M3GNetMLP,
    ActivationFunction,
    BondExpansion,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
)
from matgl.utils.cutoff import polynomial_cutoff



class XASGNN(nn.Module):
    """Backbone GNN that encodes atomic structure into absorber feature embeddings."""

    def __init__(
        self,
        element_types=("O", "H", "Zn", "Cl"),
        dim_node_embedding=64,
        dim_edge_embedding=64,
        max_n=3,
        max_l=3,
        nblocks=3,
        cutoff=4.0,
        threebody_cutoff=4.0,
        units=64,
        activation_type="swish",
        dropout=0.1,
        use_smooth=False,
        use_phi=False,
    ):
        super().__init__()
        activation: nn.Module = ActivationFunction[activation_type].value()

        self.element_types = element_types
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.nblocks = nblocks

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff)
        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n, max_l=max_l, cutoff=cutoff,
            use_smooth=use_smooth, use_phi=use_phi
        )

        # Embedding 
        degree_rbf = max_n * max_l
        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            activation=activation,
        )

        # 3-body interactions + graph blocks
        self.three_body_interactions = nn.ModuleList([
            ThreeBodyInteractions(
                update_network_atom=M3GNetMLP(
                    dims=[dim_node_embedding, degree_rbf],
                    activation=nn.Sigmoid(),
                    activate_last=True,
                ),
                update_network_bond=GatedMLP(
                    in_feats=degree_rbf,
                    dims=[dim_edge_embedding],
                    use_bias=False,
                ),
            )
            for _ in range(nblocks)
        ])

        self.graph_layers = nn.ModuleList([
            M3GNetBlock(
                degree=degree_rbf,
                activation=activation,
                conv_hiddens=[units, units],
                dim_node_feats=dim_node_embedding,
                dim_edge_feats=dim_edge_embedding,
                dropout=dropout,
            )
            for _ in range(nblocks)
        ])

    def forward(self, g, state_attr=None):
        """Return absorber embeddings (B, dim_node_embedding)."""
        node_types = g.ndata["node_type"]

        # Bond features
        expanded_dists = self.bond_expansion(g.edata["bond_dist"])
        l_g = create_line_graph(g, self.threebody_cutoff)
        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_basis = self.basis_expansion(l_g)
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)

        # Initial embeddings
        node_feat, edge_feat, state_feat = self.embedding(node_types, g.edata["rbf"], state_attr)

        # Message passing
        for i in range(self.nblocks):
            edge_feat = self.three_body_interactions[i](
                g, l_g, three_body_basis, three_body_cutoff, node_feat, edge_feat
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)

        # Extract absorber embeddings (first atom of each graph)
        num_atoms_per_graph = g.batch_num_nodes().tolist()
        first_atom_indices = torch.cumsum(
            torch.tensor([0] + num_atoms_per_graph[:-1], device=node_feat.device),
            dim=0
        )
        absorber_feats = node_feat[first_atom_indices]  # (B, dim_node_embedding)

        return absorber_feats


class SpectrumHead(nn.Module):
    """MLP head for spectra prediction."""
    def __init__(self, input_size=64, hidden_dims=[64, 64], output_size=200, drop_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_size] + hidden_dims + [output_size]
        for i, (w1, w2) in enumerate(itertools.pairwise(dims)):
            if i < len(dims) - 2:  # hidden
                self.layers.append(nn.Linear(w1, w2))
                self.layers.append(nn.BatchNorm1d(w2))
                self.layers.append(nn.SiLU())
                self.layers.append(nn.Dropout(drop_rate))
            else:                  # output
                self.layers.append(nn.Linear(w1, w2))
                self.layers.append(nn.Softplus())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x        

  
# class XASModel(nn.Module):
#     def __init__(self, gnn: XASGNN, head: SpectrumHead):
#         super().__init__()
#         self.gnn = gnn
#         self.head = head

#     def forward(self, g):
#         feats = self.gnn(g)      # absorber embeddings
#         spectra = self.head(feats)  # spectra
#         return spectra      


class XASLightningModule(pl.LightningModule):
    """
    Lightning wrapper for XASModel.
    Handles training, validation, and optimizer configuration.
    """

    def __init__(self, gnn_config, head_config, learning_rate=1e-3):
        super().__init__()

        # Store configs
        self.gnn_config = gnn_config
        self.head_config = head_config
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        # Build models
        self.gnn = XASGNN(**gnn_config)
        self.spectrum_head = SpectrumHead(**head_config)

        # Save hyperparameters 
        self.save_hyperparameters()

    def forward(self, g):
        feats = self.gnn(g)                 # absorber embeddings (B, d)
        spectra = self.spectrum_head(feats)    # spectra (B, output_size)
        return spectra 
        
    def training_step(self, batch, batch_idx):
        g, _, spectra = batch  # (graph, _, target_spectrum)
        g, spectra = g.to(self.device), spectra.to(self.device)
        preds = self(g)
        loss = self.loss_fn(preds, spectra)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        g, _, spectra = batch
        g, spectra = g.to(self.device), spectra.to(self.device)
        preds = self(g)
        loss = self.loss_fn(preds, spectra)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MLPLightningModule(pl.LightningModule):
    def __init__(self, head_config, learning_rate=1e-3):
        super().__init__()
        
        self.head_config = head_config
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        self.spectrum_head = SpectrumHead(**head_config)

        self.save_hyperparameters()

    def forward(self, x):
        return self.spectrum_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



"""Graph Convolutional Network for many-to-many fault classification.

Uses PyTorch Geometric's GCNConv layers to aggregate information across
sensor nodes at each timestep, followed by a temporal LSTM for sequence
modelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from DiFD.models.base import BaseModel


class GCNClassifier(BaseModel):
    """GCN model for many-to-many sequence classification on sensor graphs.

    Requires ``"graph"`` metadata (adjacency matrix and node mapping)
    to be present in the prepared dataset.

    Architecture:
        Input (batch, seq_len, num_nodes * features_per_node)
        -> reshape to (batch * seq_len * num_nodes, features_per_node)
        -> GCNConv layers with ReLU + dropout
        -> reshape to (batch, seq_len, num_nodes * gcn_hidden)
        -> LSTM temporal encoder
        -> linear projection to num_classes
        -> output (batch, seq_len, num_classes)

    The adjacency is stored as a COO edge_index buffer derived from the
    dense adjacency matrix supplied at construction.

    Args:
        input_size: Total input features per timestep (num_nodes * features_per_node).
        num_nodes: Number of graph nodes (sensors).
        adjacency: Dense adjacency matrix as list[list[float]] (with self-loops).
        gcn_hidden: Hidden dimension for GCNConv layers.
        num_gcn_layers: Number of stacked GCNConv layers.
        lstm_hidden: Hidden dimension for temporal LSTM.
        num_lstm_layers: Number of LSTM layers.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        adjacency: list[list[float]] | None = None,
        gcn_hidden: int = 64,
        num_gcn_layers: int = 2,
        lstm_hidden: int = 64,
        num_lstm_layers: int = 1,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.gcn_hidden_size = gcn_hidden
        self.num_gcn_layers = num_gcn_layers
        self.lstm_hidden_size = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout

        self.features_per_node = input_size // num_nodes

        if adjacency is not None:
            adj_tensor = torch.tensor(adjacency, dtype=torch.float32)
        else:
            adj_tensor = torch.eye(num_nodes, dtype=torch.float32)

        edge_index = adj_tensor.nonzero(as_tuple=False).t().contiguous()
        self.register_buffer("edge_index", edge_index)

        self._adjacency_list: list[list[float]] = (
            adjacency if adjacency is not None else torch.eye(num_nodes).tolist()
        )

        convs: list[GCNConv] = []
        in_dim = self.features_per_node
        for _ in range(num_gcn_layers):
            convs.append(GCNConv(in_dim, gcn_hidden))
            in_dim = gcn_hidden
        self.convs = nn.ModuleList(convs)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=gcn_hidden * num_nodes,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    @property
    def name(self) -> str:
        return "gcn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for many-to-many graph classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
                where input_size = num_nodes * features_per_node.

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        batch, seq_len, _ = x.shape
        N = self.num_nodes

        x = x.view(batch * seq_len, N, self.features_per_node)

        edge_index: torch.Tensor = self.edge_index  # type: ignore[assignment]

        offsets = torch.arange(
            batch * seq_len, device=x.device
        ).unsqueeze(1) * N
        batched_edge_index = (
            edge_index.unsqueeze(0) + offsets.unsqueeze(1)
        )
        batched_edge_index = batched_edge_index.view(2, -1)

        h = x.view(batch * seq_len * N, self.features_per_node)

        for conv in self.convs:
            h = conv(h, batched_edge_index)
            h = self.relu(h)
            h = self.dropout(h)

        h = h.view(batch, seq_len, N * self.gcn_hidden_size)

        lstm_out, _ = self.lstm(h)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits

    def get_config(self) -> dict[str, object]:
        """Return model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "adjacency": self._adjacency_list,
            "gcn_hidden": self.gcn_hidden_size,
            "num_gcn_layers": self.num_gcn_layers,
            "lstm_hidden": self.lstm_hidden_size,
            "num_lstm_layers": self.num_lstm_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> GCNClassifier:
        """Load model from a saved directory."""
        directory = Path(path)
        meta = BaseModel.load_metadata(directory)
        config = meta["model_config"]
        assert isinstance(config, dict)
        model = cls(
            input_size=int(config["input_size"]),
            num_nodes=int(config["num_nodes"]),
            adjacency=config.get("adjacency"),  # type: ignore[arg-type]
            gcn_hidden=int(config.get("gcn_hidden", 64)),
            num_gcn_layers=int(config.get("num_gcn_layers", 2)),
            lstm_hidden=int(config.get("lstm_hidden", 64)),
            num_lstm_layers=int(config.get("num_lstm_layers", 1)),
            num_classes=int(config["num_classes"]),
            dropout=float(config.get("dropout", 0.2)),
        )
        model.load_state_dict(
            torch.load(directory / "weight.pt", weights_only=True)
        )
        return model

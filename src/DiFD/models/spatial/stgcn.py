"""Spatio-Temporal Graph Convolutional Network for per-node fault classification.

Implements an ST-GCN architecture inspired by Yu et al. (2018) "Spatio-Temporal
Graph Convolutional Networks".  Each ST-Conv block applies:

    1. Temporal convolution (1-D conv along the time axis per node)
    2. Spatial graph convolution (GCNConv across nodes per timestep)
    3. Temporal convolution (1-D conv along the time axis per node)

Stacking multiple ST-Conv blocks captures multi-scale spatio-temporal
dependencies.  A final per-node linear head produces class logits for
every node at every timestep, enabling per-sensor fault diagnosis without
collapsing information across nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from DiFD.models.base import BaseModel


class TemporalConv(nn.Module):
    """Causal 1-D temporal convolution applied independently per node.

    Operates on ``(batch, num_nodes, channels, seq_len)`` tensors,
    treating each node channel slice as a 1-D signal along time.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Temporal kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, channels, num_nodes, seq_len)``.

        Returns:
            ``(batch, out_channels, num_nodes, seq_len)``.
        """
        return self.bn(self.conv(x))


class STConvBlock(nn.Module):
    """Spatio-Temporal Convolutional Block.

    Sandwich structure: TemporalConv → GCNConv → TemporalConv with
    residual connection, ReLU activations, and dropout.

    Args:
        in_channels: Input feature channels per node.
        hidden_channels: Intermediate channel dimension.
        out_channels: Output feature channels per node.
        kernel_size: Temporal convolution kernel size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.temporal1 = TemporalConv(in_channels, hidden_channels, kernel_size)
        self.gcn = GCNConv(hidden_channels, hidden_channels)
        self.temporal2 = TemporalConv(hidden_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, channels, num_nodes, seq_len)``.
            edge_index: Graph edge index ``(2, num_edges)`` for a single graph.

        Returns:
            ``(batch, out_channels, num_nodes, seq_len)``.
        """
        residual = self.residual(x)

        h = self.relu(self.temporal1(x))

        batch, C, N, T = h.shape
        h = h.permute(0, 3, 2, 1).reshape(batch * T, N, C)

        offsets = (
            torch.arange(batch * T, device=h.device).unsqueeze(1) * N
        )
        batched_ei = edge_index.unsqueeze(0) + offsets.unsqueeze(1)
        batched_ei = batched_ei.reshape(2, -1)

        h = h.reshape(batch * T * N, C)
        h = self.gcn(h, batched_ei)
        h = self.relu(h)

        h = h.reshape(batch, T, N, -1).permute(0, 3, 2, 1)

        h = self.relu(self.temporal2(h))
        h = self.dropout(h + residual)
        return h


class STGCNClassifier(BaseModel):
    """ST-GCN model for per-node many-to-many fault classification.

    Requires ``"graph"`` metadata (adjacency matrix and node mapping)
    to be present in the prepared dataset.

    Architecture::

        Input  (batch, seq_len, num_nodes * features_per_node)
          → reshape to (batch, features_per_node, num_nodes, seq_len)
          → ST-Conv blocks  (temporal conv → GCN → temporal conv) × N
          → per-node linear head
          → output (batch, seq_len, num_nodes, num_classes)

    Each sensor node gets its own classification at every timestep,
    preserving full spatial resolution for per-sensor fault diagnosis.

    Args:
        input_size: Total input features per timestep
            (``num_nodes * features_per_node``).
        num_nodes: Number of graph nodes (sensors).
        adjacency: Dense adjacency matrix as ``list[list[float]]``.
        st_hidden: Hidden channel dimension inside ST-Conv blocks.
        num_st_blocks: Number of stacked ST-Conv blocks.
        temporal_kernel: Kernel size for temporal convolutions.
        num_classes: Number of output fault classes.
        dropout: Dropout probability.
    """

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        adjacency: list[list[float]] | None = None,
        st_hidden: int = 64,
        num_st_blocks: int = 2,
        temporal_kernel: int = 3,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.st_hidden_size = st_hidden
        self.num_st_blocks = num_st_blocks
        self.temporal_kernel_size = temporal_kernel
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

        blocks: list[STConvBlock] = []
        in_ch = self.features_per_node
        for _ in range(num_st_blocks):
            blocks.append(
                STConvBlock(
                    in_channels=in_ch,
                    hidden_channels=st_hidden,
                    out_channels=st_hidden,
                    kernel_size=temporal_kernel,
                    dropout=dropout,
                )
            )
            in_ch = st_hidden
        self.blocks = nn.ModuleList(blocks)

        self.fc = nn.Linear(st_hidden, num_classes)

    @property
    def name(self) -> str:
        return "stgcn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for per-node many-to-many classification.

        Args:
            x: Input tensor ``(batch, seq_len, input_size)`` where
                ``input_size = num_nodes * features_per_node``.

        Returns:
            Logits tensor ``(batch, seq_len, num_nodes, num_classes)``.
        """
        batch, seq_len, _ = x.shape
        N = self.num_nodes
        F_in = self.features_per_node

        h = x.view(batch, seq_len, N, F_in)
        h = h.permute(0, 3, 2, 1)

        edge_index: torch.Tensor = self.edge_index  # type: ignore[assignment]

        for block in self.blocks:
            h = block(h, edge_index)

        h = h.permute(0, 3, 2, 1)

        logits = self.fc(h)
        return logits

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "adjacency": self._adjacency_list,
            "st_hidden": self.st_hidden_size,
            "num_st_blocks": self.num_st_blocks,
            "temporal_kernel": self.temporal_kernel_size,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> STGCNClassifier:
        """Load model from a saved directory."""
        directory = Path(path)
        meta = BaseModel.load_metadata(directory)
        config = meta["model_config"]
        assert isinstance(config, dict)
        model = cls(
            input_size=int(config["input_size"]),
            num_nodes=int(config["num_nodes"]),
            adjacency=config.get("adjacency"),  # type: ignore[arg-type]
            st_hidden=int(config.get("st_hidden", 64)),
            num_st_blocks=int(config.get("num_st_blocks", 2)),
            temporal_kernel=int(config.get("temporal_kernel", 3)),
            num_classes=int(config["num_classes"]),
            dropout=float(config.get("dropout", 0.2)),
        )
        model.load_state_dict(
            torch.load(directory / "weight.pt", weights_only=True)
        )
        return model

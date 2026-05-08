"""CESTA spatial-temporal model for communication-aware fault diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, TypedDict, cast

import torch
import torch.nn as nn

from CADFD.models.base import BaseModel

CommunicationMode = Literal["none", "dense"]


class CommunicationStats(TypedDict):
    """Communication statistics captured from the most recent forward pass."""

    active_request_ratio: float
    requested_edge_count: float
    possible_edge_count: float
    transmitted_bits_estimate: float
    full_embedding_message_count: float
    compressed_message_count: float
    average_compression_ratio: float


class CESTAClassifier(BaseModel):
    """Communication-Efficient Spatial-Temporal Aggregation classifier."""

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        adjacency: list[list[float]] | None = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 4,
        dropout: float = 0.2,
        communication_mode: CommunicationMode = "none",
        fusion_hidden_size: int | None = None,
        precision_bits: int = 32,
    ) -> None:
        super().__init__()
        if input_size % num_nodes != 0:
            raise ValueError("input_size must be divisible by num_nodes")
        if communication_mode not in {"none", "dense"}:
            raise ValueError("communication_mode must be one of: none, dense")

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.features_per_node = input_size // num_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.communication_mode: CommunicationMode = communication_mode
        self.fusion_hidden_size = fusion_hidden_size
        self.precision_bits = precision_bits

        if adjacency is not None:
            adj_tensor = torch.tensor(adjacency, dtype=torch.float32)
        else:
            adj_tensor = torch.eye(num_nodes, dtype=torch.float32)
        if adj_tensor.shape != (num_nodes, num_nodes):
            raise ValueError("adjacency must have shape (num_nodes, num_nodes)")
        self.register_buffer("adjacency", adj_tensor)
        self._adjacency_list: list[list[float]] = adj_tensor.tolist()

        self.temporal_encoder = nn.GRU(
            input_size=self.features_per_node,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, fusion_hidden_size or hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size or hidden_size, hidden_size),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._last_communication_stats: CommunicationStats = (
            self._zero_communication_stats()
        )

    @property
    def name(self) -> str:
        return "cesta"

    @property
    def last_communication_stats(self) -> CommunicationStats:
        return self._last_communication_stats.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        local_input = x.view(batch, seq_len, self.num_nodes, self.features_per_node)
        local_input = local_input.permute(0, 2, 1, 3).reshape(
            batch * self.num_nodes, seq_len, self.features_per_node
        )

        local_hidden, _ = self.temporal_encoder(local_input)
        local_hidden = local_hidden.view(
            batch, self.num_nodes, seq_len, self.hidden_size
        )
        local_hidden = local_hidden.permute(0, 2, 1, 3)

        if self.communication_mode == "dense":
            neighbor_context = self._dense_neighbor_context(local_hidden)
            fused = self.fusion(torch.cat([local_hidden, neighbor_context], dim=-1))
            hidden = self.dropout(local_hidden + fused)
            self._last_communication_stats = self._dense_communication_stats(
                batch=batch,
                seq_len=seq_len,
                device=x.device,
            )
        else:
            hidden = self.dropout(local_hidden)
            self._last_communication_stats = self._zero_communication_stats()

        return self.classifier(hidden)

    def _dense_neighbor_context(self, local_hidden: torch.Tensor) -> torch.Tensor:
        adjacency = cast(torch.Tensor, self.adjacency)
        message_mask = adjacency.clone()
        message_mask.fill_diagonal_(0.0)
        degree = message_mask.sum(dim=1).clamp_min(1.0)
        normalized = message_mask / degree.unsqueeze(1)
        return torch.einsum("ij,btjh->btih", normalized, local_hidden)

    def _zero_communication_stats(self) -> CommunicationStats:
        return {
            "active_request_ratio": 0.0,
            "requested_edge_count": 0.0,
            "possible_edge_count": self._possible_edge_count(),
            "transmitted_bits_estimate": 0.0,
            "full_embedding_message_count": 0.0,
            "compressed_message_count": 0.0,
            "average_compression_ratio": 0.0,
        }

    def _dense_communication_stats(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> CommunicationStats:
        possible_edges = self._possible_edge_count(device=device)
        requested_edges = possible_edges * batch * seq_len
        transmitted_bits = requested_edges * self.hidden_size * self.precision_bits
        return {
            "active_request_ratio": 1.0 if possible_edges > 0 else 0.0,
            "requested_edge_count": float(requested_edges),
            "possible_edge_count": float(possible_edges * batch * seq_len),
            "transmitted_bits_estimate": float(transmitted_bits),
            "full_embedding_message_count": float(requested_edges),
            "compressed_message_count": 0.0,
            "average_compression_ratio": 1.0 if possible_edges > 0 else 0.0,
        }

    def _possible_edge_count(self, device: torch.device | None = None) -> float:
        adjacency = cast(torch.Tensor, self.adjacency)
        if device is not None:
            adjacency = adjacency.to(device)
        message_mask = adjacency.clone()
        message_mask.fill_diagonal_(0.0)
        return float(message_mask.sum().item())

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "adjacency": self._adjacency_list,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "communication_mode": self.communication_mode,
            "fusion_hidden_size": self.fusion_hidden_size,
            "precision_bits": self.precision_bits,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> CESTAClassifier:
        directory = Path(path)
        meta = BaseModel.load_metadata(directory)
        config = meta["model_config"]
        assert isinstance(config, dict)
        model = cls(
            input_size=int(config["input_size"]),
            num_nodes=int(config["num_nodes"]),
            adjacency=config.get("adjacency"),  # type: ignore[arg-type]
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 1)),
            num_classes=int(config["num_classes"]),
            dropout=float(config.get("dropout", 0.2)),
            communication_mode=config.get("communication_mode", "none"),  # type: ignore[arg-type]
            fusion_hidden_size=(
                int(config["fusion_hidden_size"])
                if config.get("fusion_hidden_size") is not None
                else None
            ),
            precision_bits=int(config.get("precision_bits", 32)),
        )
        model.load_state_dict(torch.load(directory / "weight.pt", weights_only=True))
        return model

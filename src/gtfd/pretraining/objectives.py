"""Self-supervised pretraining objectives for GTFD."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import negative_sampling

from ..config import PretrainingConfig


class LinkMaskLoss(nn.Module):
    """Link masking loss for self-supervised pretraining.

    Masks a fraction of edges and predicts their existence and attributes.
    """

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.mask_ratio = config.mask_ratio

    def forward(
        self,
        model: nn.Module,
        data: Data | HeteroData,
        h_fused: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute link masking loss.

        Args:
            model: GTFD model with edge predictor head.
            data: Input graph data.
            h_fused: Fused node embeddings.

        Returns:
            Total loss and loss components dict.
        """
        if isinstance(data, HeteroData):
            return self._forward_hetero(model, data, h_fused)
        return self._forward_homo(model, data, h_fused)

    def _forward_homo(
        self,
        model: nn.Module,
        data: Data,
        h_fused: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Homogeneous graph link masking."""
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        num_mask = int(num_edges * self.mask_ratio)

        # Randomly mask edges
        perm = torch.randperm(num_edges, device=edge_index.device)
        mask_idx = perm[:num_mask]
        keep_idx = perm[num_mask:]

        pos_edge_index = edge_index[:, mask_idx]
        neg_edge_index = negative_sampling(
            edge_index=edge_index[:, keep_idx],
            num_nodes=data.num_nodes,
            num_neg_samples=num_mask,
            method="sparse",
        )

        # Predict edge existence
        pos_pred = model.predict_edges(h_fused, pos_edge_index)
        neg_pred = model.predict_edges(h_fused, neg_edge_index)

        pos_labels = torch.ones(pos_pred.shape[0], device=pos_pred.device)
        neg_labels = torch.zeros(neg_pred.shape[0], device=neg_pred.device)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = F.binary_cross_entropy_with_logits(pred, labels)

        # Edge attribute prediction (if available)
        attr_loss = torch.tensor(0.0, device=loss.device)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            pos_edge_attr = data.edge_attr[mask_idx]
            attr_pred = model.predict_edge_attrs(h_fused, pos_edge_index)
            attr_loss = F.mse_loss(attr_pred, pos_edge_attr)

        total_loss = loss + 0.1 * attr_loss
        return total_loss, {
            "link_mask_loss": loss.detach(),
            "attr_loss": attr_loss.detach(),
        }

    def _forward_hetero(
        self,
        model: nn.Module,
        data: HeteroData,
        h_fused: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Heterogeneous graph link masking."""
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        losses = {}

        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            num_edges = edge_index.shape[1]
            if num_edges == 0:
                continue

            num_mask = int(num_edges * self.mask_ratio)
            perm = torch.randperm(num_edges, device=edge_index.device)
            mask_idx = perm[:num_mask]
            keep_idx = perm[num_mask:]

            pos_edge_index = edge_index[:, mask_idx]
            neg_edge_index = negative_sampling(
                edge_index=edge_index[:, keep_idx],
                num_nodes=(data[edge_type[0]].num_nodes, data[edge_type[2]].num_nodes),
                num_neg_samples=num_mask,
                method="sparse",
            )

            src_type, _, dst_type = edge_type
            pos_pred = model.predict_edges(h_fused[src_type], h_fused[dst_type], pos_edge_index)
            neg_pred = model.predict_edges(h_fused[src_type], h_fused[dst_type], neg_edge_index)

            pos_labels = torch.ones(pos_pred.shape[0], device=pos_pred.device)
            neg_labels = torch.zeros(neg_pred.shape[0], device=neg_pred.device)

            pred = torch.cat([pos_pred, neg_pred], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            loss = F.binary_cross_entropy_with_logits(pred, labels)
            total_loss = total_loss + loss
            losses[f"link_mask_loss_{edge_type[1]}"] = loss.detach()

        return total_loss, losses


class AdversarialLoss(nn.Module):
    """Adversarial loss for robust pretraining.

    Maximizes loss on adversarially perturbed graphs.
    """

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        model: nn.Module,
        data: Data | HeteroData,
        h_fused: torch.Tensor | dict[str, torch.Tensor],
        h_adv: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute adversarial loss.

        Args:
            model: GTFD model.
            data: Clean graph data.
            h_fused: Clean fused embeddings.
            h_adv: Adversarially perturbed embeddings.

        Returns:
            Adversarial loss and components.
        """
        # Consistency loss between clean and adversarial embeddings
        if isinstance(h_fused, dict):
            loss = 0.0
            for node_type in h_fused:
                if node_type in h_adv:
                    loss = loss + F.mse_loss(h_fused[node_type], h_adv[node_type])
            loss = loss / len(h_fused)
        else:
            loss = F.mse_loss(h_fused, h_adv)

        return loss, {"adv_loss": loss.detach()}


class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning.

    InfoNCE loss between clean and augmented views.
    """

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature

    def forward(
        self,
        h_clean: torch.Tensor | dict[str, torch.Tensor],
        h_aug: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute contrastive loss.

        Args:
            h_clean: Clean embeddings.
            h_aug: Augmented embeddings.

        Returns:
            Contrastive loss and components.
        """
        if isinstance(h_clean, dict):
            total_loss = 0.0
            losses = {}
            for node_type in h_clean:
                if node_type in h_aug:
                    loss = self._info_nce(h_clean[node_type], h_aug[node_type])
                    total_loss = total_loss + loss
                    losses[f"contrastive_{node_type}"] = loss.detach()
            return total_loss / len(losses) if losses else torch.tensor(0.0), losses

        return self._info_nce(h_clean, h_aug), {"contrastive_loss": self._info_nce(h_clean, h_aug).detach()}

    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE loss between two views."""
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Similarity matrix
        sim = torch.mm(z1, z2.t()) / self.temperature  # [batch, batch]

        # Positive pairs are on diagonal
        labels = torch.arange(batch_size, device=z1.device, dtype=torch.long)

        loss = F.cross_entropy(sim, labels)
        return loss


class PretrainingObjectives(nn.Module):
    """Combined pretraining objectives for GTFD.

    Combines link masking, adversarial, and contrastive losses.
    """

    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.link_mask = LinkMaskLoss(config)
        self.adversarial = AdversarialLoss(config)
        self.contrastive = ContrastiveLoss(config)

    def forward(
        self,
        model: nn.Module,
        data: Data | HeteroData,
        h_fused: torch.Tensor | dict[str, torch.Tensor],
        h_adv: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
        h_aug: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined pretraining loss.

        Args:
            model: GTFD model.
            data: Input graph data.
            h_fused: Clean fused embeddings.
            h_adv: Adversarial embeddings (optional).
            h_aug: Augmented embeddings (optional).

        Returns:
            Total loss and loss components.
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        # Link masking loss
        link_loss, link_losses = self.link_mask(model, data, h_fused)
        total_loss = total_loss + self.config.lambda_link * link_loss
        losses.update(link_losses)

        # Adversarial loss
        if h_adv is not None and self.config.lambda_adv > 0:
            adv_loss, adv_losses = self.adversarial(model, data, h_fused, h_adv)
            total_loss = total_loss + self.config.lambda_adv * adv_loss
            losses.update(adv_losses)

        # Contrastive loss
        if h_aug is not None and self.config.lambda_contrast > 0:
            cont_loss, cont_losses = self.contrastive(h_fused, h_aug)
            total_loss = total_loss + self.config.lambda_contrast * cont_loss
            losses.update(cont_losses)

        losses["total_loss"] = total_loss.detach()
        return total_loss, losses
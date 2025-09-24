# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Implementation of the paper Protecting Split Learning by Potential Energy Loss:
# https://www.ijcai.org/proceedings/2024/0618.pdf.


import types
import logging

import torch
from torch import nn
import torch.nn.functional as F

from sfl.ml.nn.callbacks.callback import Callback
from sfl.ml.nn.core.torch import module
from sfl.ml.nn.callbacks.callback import Callback
from sfl.ml.nn.core.torch import loss_wrapper, module, TorchModel
from sfl.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel


class PELoss(nn.Module):
    """Potential Energy Loss(PELoss) From Paper:
    'Protecting Split Learning by Potential Energy Loss'"""

    def __init__(self, alpha=2.0, use_angular=False):
        """
        Args:
            alpha: weight of PE loss
            use_angular: whether to use angular distance.
        MENTION: angular distance perform loss explosion in some cases.
        """
        super().__init__()
        self.alpha = alpha
        self.use_angular = use_angular
        self.eps = 1e-8
        self.cce_loss = torch.nn.CrossEntropyLoss()


    def forward(self, embeddings, preds, labels):
        total_loss = 0.0
        if isinstance(embeddings, list):
            embeddings = embeddings[0]
        unique_labels = torch.unique(labels)

        for label in unique_labels:

            mask = labels == label
            class_emb = embeddings[mask]
            if class_emb.shape[0] < 2:
                continue
            if self.use_angular:
                class_emb = F.layer_norm(class_emb, class_emb.shape[1:])
                class_emb = F.normalize(class_emb, dim=1, p=2)
                cos_sim = class_emb @ class_emb.T  # cosine similarity
                dist_matrix = torch.arccos(
                    torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)
                )
            else:
                dist_matrix = torch.cdist(class_emb, class_emb, p=2)

            mask = ~torch.eye(
                class_emb.shape[0], device=embeddings.device, dtype=torch.bool
            )
            dist_matrix = dist_matrix[mask]
            loss = torch.sum(1.0 / (dist_matrix) + self.eps)
            total_loss += loss / class_emb.shape[0]
        cce_loss = self.cce_loss(preds, labels)
        return self.alpha * total_loss + cce_loss if unique_labels.numel() > 0 else 0.0


class DcorLoss(nn.Module):
    """Distance Correlation Loss (DcorLoss) From Paper: 'NoPeek: Information
    leakage reduction to share activations in distributed deep learning' and
    'Protecting Split Learning by Potential Energy Loss'"""

    def __init__(self, num_classes=10, dcor_weighting: float = 0.1) -> None:

        super().__init__()
        self.num_classes = num_classes
        self.cce_loss = torch.nn.CrossEntropyLoss()
        self.dcor_weighting = dcor_weighting
        logging.info(
            f'DcorLoss initialized with num_classes={self.num_classes}, dcor_weighting={self.dcor_weighting}'
        )
        self.eps = 1e-8

    def forward(self, embeddings, preds, labels):
        if isinstance(embeddings, list):
            embeddings = embeddings[0]
        if labels.type() != torch.int64:
            labels = labels.long()
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes)
        labels_onehot = labels_onehot.float()

        dist_emb = self._distance_matrix(embeddings)
        dist_label = self._distance_matrix(labels_onehot)
        dist_emb = self._double_center(dist_emb)
        dist_label = self._double_center(dist_label)

        dcov = torch.mean(dist_emb * dist_label)
        dvar_emb = torch.mean(dist_emb**2)
        dvar_label = torch.mean(dist_label**2)

        dcor = dcov / (torch.sqrt(dvar_emb * dvar_label) + self.eps)
        return self.dcor_weighting * dcor + self.cce_loss(preds, labels)

    def _distance_matrix(self, x):
        return torch.cdist(x, x, p=2)

    def _double_center(self, mat):
        row_mean = mat.mean(dim=1, keepdim=True)
        col_mean = mat.mean(dim=0, keepdim=True)
        total_mean = mat.mean()
        return mat - row_mean - col_mean + total_mean


def loss_wrapper(loss_type, **kwargs):
    def wrapper():
        if loss_type == 'peloss':
            loss_func = PELoss(**kwargs)
        elif loss_type == 'dcorloss':
            loss_func = DcorLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        return loss_func

    return wrapper


class BaselossDefense(Callback):
    """Base class for loss-based defenses. Can be modified to implement specific loss-based defenses."""

    def __init__(self, loss_type, alpha, use_angular=False, num_classes=10, **kwargs):
        self.loss_type = loss_type
        self.alpha = alpha
        self.use_angular = use_angular
        self.num_classes = num_classes
        super().__init__(**kwargs)

    @staticmethod
    def inject_forward_method(worker: SLBaseTorchModel, loss_func):
        def self_defined_forward_step(
            self, batch, batch_idx: int, dataloader_idx: int = 0
        ):
            x, y = batch
            y_pred = self(x)

            self.update_metrics(y_pred, y)

            if self.loss:
                loss = self.loss(x, y_pred, y)
                return y_pred, loss
            else:
                return y_pred, None

        # rebuild model_fuse with new loss function.
        # MENTION: The cnn defined not as another model.
        # due to the loss function init in the TorchModel, the arguments of the loss hard to modify.
        # Or you can use the decorator to wrap the loss function.
        worker.builder_fuse = TorchModel(
            model_fn=worker.builder_fuse.model_fn,
            optim_fn=worker.builder_fuse.optim_fn,
            loss_fn=loss_func,
            metrics=worker.builder_fuse.metrics,
            **worker.builder_fuse.kwargs,
        )
        worker.model_fuse = module.build(worker.builder_fuse, worker.exec_device)

        worker.model_fuse.forward_step = types.MethodType(
            self_defined_forward_step, worker.model_fuse
        )

    def on_train_begin(self, logs=None):
        if self.loss_type == "peloss":
            loss_func = loss_wrapper(
                self.loss_type, alpha=self.alpha, use_angular=self.use_angular
            )
        elif self.loss_type == "dcorloss":
            loss_func = loss_wrapper(
                self.loss_type, num_classes=self.num_classes, dcor_weighting=self.alpha
            )
        worker = self._workers[self.device_y]
        worker.apply(self.inject_forward_method, loss_func)
# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import logging
from secretflow.data.ndarray import FedNdarray, PartitionWay
from sfl.ml.nn import SLModel
from sfl.ml.nn.core.torch import TorchModel, metric_wrapper, optim_wrapper
from sfl.ml.nn.sl.attacks.sim_lia_torch import SimilarityLabelInferenceAttack
from sfl.ml.nn.sl.defenses.loss_defense import BaselossDefense


def do_test_sl_and_sim_lia(alice, bob, config):

    class BaseNet(nn.Module):
        def __init__(self):
            super(BaseNet, self).__init__()
            self.fc1 = nn.Linear(100, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 100)
            self.ReLU = nn.ReLU()

        def forward(self, x):

            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.fc2(x)
            x = self.ReLU(x)
            x = self.fc3(x)
            return x

        def output_num(self):
            return 1

    class FuseNet(nn.Module):
        def __init__(self):
            super(FuseNet, self).__init__()
            self.fc1 = nn.Linear(100, 100)
            self.fc2 = nn.Linear(100, 10)
            self.ReLU = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.fc2(x)
            return x

    train_data = np.random.rand(1000, 100).astype(np.float32)
    train_label = np.random.randint(0, 10, size=(1000,)).astype(np.int64)

    # put into FedNdarray
    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(train_data),
            # bob: bob(lambda x: x)(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    label = bob(lambda x: x)(train_label)

    # model configure
    loss_fn = nn.CrossEntropyLoss

    optim_fn = optim_wrapper(optim.SGD, lr=1e-2, momentum=0.9, weight_decay=5e-4)

    _base_model, _fuse_model = BaseNet, FuseNet

    base_model = TorchModel(
        model_fn=_base_model,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    fuse_model = TorchModel(
        model_fn=_fuse_model,
        loss_fn=loss_fn,
        optim_fn=optim_fn,
        metrics=[
            metric_wrapper(
                Accuracy, task="multiclass", num_classes=10, average="micro"
            ),
        ],
    )

    base_model_dict = {
        alice: base_model,
    }

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=bob,
        model_fuse=fuse_model,
        simulation=True,
        backend="torch",
        strategy="split_nn",
    )

    loss_type, alpha = config.split(",")
    print(loss_type)
    if loss_type == "peloss":
        loss_defense = BaselossDefense(
            loss_type='peloss', alpha=int(alpha), use_angular=False
        )
    elif loss_type == "dcorloss":
        loss_defense = BaselossDefense(
            loss_type='dcorloss', alpha=int(alpha), num_classes=10
        )
    else:
        raise ValueError(f"Unsupported loss type: {config}")

    history = sl_model.fit(
        fed_data,
        label,
        validation_data=(fed_data, label),
        epochs=2,
        batch_size=128,
        random_seed=1234,
        callbacks=[loss_defense],
    )
    print(history)

    pred_bs = 128
    result = sl_model.predict(fed_data, batch_size=pred_bs, verbose=1)

    return result


def test_sl_and_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    config = []
    for i in range(1, 5, 2):
        config.append(f"peloss,{2**i}")
        config.append(f"dcorloss,{2**i}")
    for i in config:
        do_test_sl_and_sim_lia(alice, bob, i)

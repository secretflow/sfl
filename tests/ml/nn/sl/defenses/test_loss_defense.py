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

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n_samples=2000, n_features=100, n_classes=10):
    """
    生成具有可学习模式的合成数据
    参数:
        n_samples: 样本数量
        n_features: 特征维度
        n_classes: 类别数量
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=80,
        n_redundant=10, 
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        flip_y=0.01,
        random_state=42
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X.astype(np.float32), y.astype(np.int64)

def do_test_sl_and_sim_lia(alice, bob, config):

    class BaseNet(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=128, output_dim=64):
            super(BaseNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.bn3 = nn.BatchNorm1d(output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.bn3(self.fc3(x)) 
            return x
        
        def output_num(self):
            return 1

    class FuseNet(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=32, n_classes=10):
            super(FuseNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x) 
            return x

    data, label = generate_synthetic_data(n_samples=2500, n_features=100, n_classes=10)
    train_data, eval_data = data[:2000], data[2000:]
    train_label, eval_label = label[:2000], label[2000:]
    # put into FedNdarray
    fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(train_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label = bob(lambda x: x)(train_label)

    eval_fed_data = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(eval_data),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    eval_label = bob(lambda x: x)(eval_label)

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

    loss_type, alpha, defense = config.split(",")
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
    
    sim_lia_callback = SimilarityLabelInferenceAttack(
        attack_party=alice,
        label_party=bob,
        data_type="grad",
        attack_method="distance",
        known_num=10,
        distance_metric="cosine",
        exec_device="cpu",
    )
    callbacks = [loss_defense] if defense == "1" else []
    callbacks.append(sim_lia_callback)

    history = sl_model.fit(
        fed_data,
        label,
        validation_data=(eval_fed_data, eval_label),
        epochs=10,
        batch_size=128,
        random_seed=1234,
        callbacks=[loss_defense],
    )
    logging.info("history:")
    logging.info(history)

    pred_bs = 128
    result = sl_model.predict(fed_data, batch_size=pred_bs, verbose=1)
    logging.info(f"predict: {config}")
    logging.info(sim_lia_callback.get_attack_metrics())
    return result


def test_sl_and_lia(sf_simulation_setup_devices):
    alice = sf_simulation_setup_devices.alice
    bob = sf_simulation_setup_devices.bob
    config = []
    for i in [2, 4]:
        for defense in [0, 1]:
            config.append(f"peloss,{2**i},{defense}")
            config.append(f"dcorloss,{2**i},{defense}")
    for i in config:
        do_test_sl_and_sim_lia(alice, bob, i)

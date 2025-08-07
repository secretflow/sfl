#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("..")

from torch import nn, optim

from . import cifar_model, gtsrb_model, mnist_model, nuswide_model

MODELS = {
    "mnist": {
        "lr": 0.01,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 80,
        # 'epochs': 1,
        "batch_size": 128,
        "model": mnist_model,
        "party_num": 2,
    },
    "nus-wide": {
        "lr": 0.0001,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 100,
        # 'epochs': 1,
        "batch_size": 128,
        "model": nuswide_model,
        "party_num": 2,
    },
    "cifar10": {
        "lr": 0.001,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 20,
        # 'epochs': 1,
        "batch_size": 128,
        "model": cifar_model,
        "party_num": 2,
    },
    "gtsrb": {
        "lr": 0.001,
        "loss": nn.CrossEntropyLoss(),
        "epochs": 20,
        # 'epochs': 1,
        "batch_size": 128,
        "model": gtsrb_model,
        "party_num": 2,
    },
}

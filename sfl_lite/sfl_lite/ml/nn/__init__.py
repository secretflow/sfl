# Copyright 2025 Ant Group Co., Ltd.
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

"""
Neural Network Split Learning Module for sfl_lite.

This module provides split learning capabilities for neural networks,
converging the split learning features from sfl to sfl_lite's MPLang architecture.
"""

from .split_learning_model import SplitLearningNN
from .base_model import (
    BaseNN,
    FuseNN,
    CNNBase,
    RNNBase,
    create_base_model,
    create_fuse_model,
)

__all__ = [
    "SplitLearningNN",
    "BaseNN",
    "FuseNN", 
    "CNNBase",
    "RNNBase",
    "create_base_model",
    "create_fuse_model",
]

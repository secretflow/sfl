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

import dataclasses
from typing import Dict, Optional

from mplang.v1.core import MPObject


@dataclasses.dataclass
class LinearRegressionModel:
    """
    Linear regression model.

    Attributes:
        weights: Dictionary mapping device names to their corresponding weight MPObject
        label_party: Device name of the party holding the labels and intercept
        intercept: Intercept (bias) term of the model as MPObject, or None if not fitted
    """

    weights: Dict[str, MPObject]
    label_party: str
    intercept: Optional[MPObject] = None

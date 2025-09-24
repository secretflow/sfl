# Copyright 2024 Ant Group Co., Ltd.
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

from typing import Dict

from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    InputMode,
    ModelType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.defenses.base import DefenseBase
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from sfl.ml.nn.callbacks.callback import Callback
from sfl.ml.nn.sl.defenses.loss_defense import (
    BaselossDefense,
)


class BaseLossDefenseFrontend(DefenseBase):
    """Base class for loss-based defenses."""

    def __str__(self):
        return "loss_based_defense"

    def build_defense_callback(self, app: ApplicationBase) -> Callback | None:
        raise NotImplementedError(
            'build_defense_callback must be implemented in subclass'
        )

    def check_attack_valid(self, attack: AttackBase) -> bool:
        return attack.attack_type() == AttackType.LABLE_INFERENCE

    def tune_metrics(self, app_metrics: Dict[str, str]) -> Dict[str, str]:
        return {}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        """only support dnn"""
        return (
            app.model_type()
            in [
                ModelType.DNN,
                ModelType.RESNET18,
                ModelType.VGG16,
                ModelType.CNN,
                ModelType.RESNET20,
            ]
            and app.base_input_mode() == InputMode.SINGLE
        )

    def update_resources_consumptions(
        self,
        cluster_resources_pack: ResourcesPack,
        app: ApplicationBase,
    ) -> ResourcesPack:
        update_gpu = lambda x: x
        update_mem = lambda x: x
        return (
            cluster_resources_pack.apply_debug_resources('gpu_mem', update_gpu)
            .apply_debug_resources('memory', update_mem)
            .apply_sim_resources(app.device_y.party, 'gpu_mem', update_gpu)
            .apply_sim_resources(app.device_y.party, 'memory', update_mem)
        )


class PELossDefense(BaseLossDefenseFrontend):

    def __str__(self):
        return "peloss_defense"

    def build_defense_callback(self, app: ApplicationBase) -> Callback | None:
        # angular loss may result in loss explosion
        return BaselossDefense(loss_type='peloss', alpha=self.config.get('alpha', 2), use_angular=False)


class DcorLossDefense(BaseLossDefenseFrontend):

    def __str__(self):
        return "dcorloss_defense"

    def build_defense_callback(self, app: ApplicationBase) -> Callback | None:
        return BaselossDefense(
            loss_type='dcorloss', alpha=self.config.get('alpha', 2), num_classes=app.num_classes
        )
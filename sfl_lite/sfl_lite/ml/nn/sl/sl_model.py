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
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import jax
import mplang.device as mpd
from flax import linen as nn, nnx
from mplang.core import MPObject


@dataclass
class SLModel:
    """SLModel"""

    # base_model: in fact nnx.Module on each party
    base_models: Dict[str, MPObject]
    fuse_model: MPObject
    label_party: str  # assumes label party holds fuse model


@mpd.function
def init_sl_model(
    base_models: Dict[str, nnx.Module], fuse_model: nnx.Module, label_party: str
) -> Tuple[Dict[str, MPObject], MPObject]:
    """init SLModel"""

    base_models_mp = {
        party: mpd.put(party, nnx.split(model)) for party, model in base_models.items()
    }
    fuse_model_mp = mpd.put(label_party, nnx.split(fuse_model))
    return base_models_mp, fuse_model_mp


@mpd.function
def functional_forward(graph_def_state, x):
    model = nnx.merge(*graph_def_state)
    y = model(x)
    _, state = nnx.split(model)
    return y, state


@mpd.function
def label_compress(base_outputs):
    return sum(base_outputs)


@mpd.function
def sl_model_forward(
    base_model_mp: Dict[str, MPObject],
    fuse_model: MPObject,
    label_party: str,
    x: Dict[str, MPObject],
) -> Tuple[MPObject, Dict[str, MPObject], MPObject]:
    """forward of model on device"""
    # 1. Call base model on each party
    base_outputs = {}
    base_states = {}
    for party, graph_def_state in base_model_mp.items():
        y, state = mpd.device(party)(functional_forward)(graph_def_state, x[party])
        base_outputs[party] = y
        base_states[party] = state

    base_outputs_label_party = [
        mpd.put(label_party, base_outputs[party]) for party in base_outputs
    ]
    compressed = mpd.device(label_party)(label_compress)(base_outputs_label_party)
    y, state = mpd.device(label_party)(functional_forward)(fuse_model, compressed)
    fuse_model_state = state
    return y, base_states, fuse_model_state


def forward_prediction_example():
    """
    完整的SL模型前向预测示例
    
    这个函数展示了如何使用SLModel进行分布式前向预测，包括：
    1. 设置多参与方计算环境
    2. 创建和初始化模型
    3. 准备输入数据
    4. 执行前向传播
    5. 获取预测结果
    
    Returns:
        dict: 包含预测结果和模型信息的字典
    """
    import mplang as mp
    import jax.numpy as jnp
    from flax import linen as nn
    from sfl_lite.ml.nn.models import ModelFactory

    # 1. 初始化模拟器
    cluster_spec = mp.ClusterSpec.from_dict(
        {
            "nodes": [
                {"name": "node_0", "endpoint": "127.0.0.1:61920"},
                {"name": "node_1", "endpoint": "127.0.0.1:61921"},
                {"name": "node_2", "endpoint": "127.0.0.1:61922"},
                {"name": "node_3", "endpoint": "127.0.0.1:61923"},
                {"name": "node_4", "endpoint": "127.0.0.1:61924"},
                {"name": "node_5", "endpoint": "127.0.0.1:61925"},
            ],
            "devices": {
                "SP0": {
                    "kind": "SPU",
                    "members": ["node_1", "node_2", "node_3"],
                    "config": {
                        "protocol": "SEMI2K",
                        "field": "FM128",
                        "enable_pphlo_profile": True,
                    },
                },
                "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
                "P1": {"kind": "PPU", "members": ["node_4"], "config": {}},
                "TEE0": {"kind": "TEE", "members": ["node_5"], "config": {}},
            },
        }
    )
    sim = mp.Simulator(cluster_spec)

    # 2. 创建模型
    rngs = nnx.Rngs(42)
    
    # 创建基础模型（每个参与方一个）
    base_model_p0 = ModelFactory.create_model("dnn", input_dim=10, num_classes=3, rngs=rngs)
    base_model_p1 = ModelFactory.create_model("dnn", input_dim=8, num_classes=3, rngs=rngs)
    
    # 创建融合模型（在标签方上）
    fuse_model = ModelFactory.create_model("dnn", input_dim=6, num_classes=3, rngs=rngs)

    # 3. 初始化SL模型
    base_model_mp, fuse_model_mp = mp.evaluate(
        sim,
        init_sl_model,
        base_models={"P0": base_model_p0, "P1": base_model_p1},
        fuse_model=fuse_model,
        label_party="P0",
    )
    sl_model = SLModel(
        base_models=base_model_mp, 
        fuse_model=fuse_model_mp, 
        label_party="P0"
    )

    # 4. 准备输入数据
    # 注意：每个参与方的输入维度可能不同
    @mpd.function
    def create_input_data(num_rows, input_dim_1, input_dim_2):
        input_p0 = jnp.ones((num_rows, input_dim_1))  # P0的输入数据
        input_p1 = jnp.ones((num_rows, input_dim_2))   # P1的输入数据
        # 将输入数据放在对应的设备上
        x_mp_0 = mpd.put("P0", input_p0)  # 主输入放在P0
        x_mp_1 = mpd.put("P1", input_p1)  # 辅助输入放在P1
        return {"P0": x_mp_0, "P1": x_mp_1}

    x_mp = mp.evaluate(sim, create_input_data, 10, 10, 8)
    # 5. 执行前向预测
    prediction, base_states, fuse_state = mp.evaluate(
        sim,
        sl_model_forward,
        base_model_mp=sl_model.base_models,
        fuse_model=sl_model.fuse_model,
        label_party=sl_model.label_party,
        x=x_mp
    )

    # 6. 获取预测结果
    result = mpd.fetch(sim, prediction)
    
    return {
        "prediction": result,
        "model_structure": {
            "base_models": list(sl_model.base_models.keys()),
            "label_party": sl_model.label_party,
            "input_shapes": {"P0": input_p0.shape, "P1": input_p1.shape},
            "output_shape": result.shape
        },
        "simulator": sim
    }


if __name__ == "__main__":
    # 运行前向预测示例
    print("=" * 50)
    print("SL模型前向预测示例")
    print("=" * 50)
    
    result = forward_prediction_example()
    print("预测结果:", result["prediction"])
    print("模型结构:", result["model_structure"])
    print("预测成功完成！")

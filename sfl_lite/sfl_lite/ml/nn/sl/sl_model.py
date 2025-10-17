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
from typing import Dict, Sequence
from mplang.core import MPObject
import mplang.device as mpd
import jax
from flax import nnx
from flax import linen as nn


@dataclass
class SLModel:
    """SLModel"""
    # base_model: in fact nnx.Module on each party
    base_models: Dict[str, MPObject]
    fuse_model: MPObject
    label_party: str # assumes label party holds fuse model

@mpd.function
def device_forward(device: str, model: MPObject, x: MPObject) -> MPObject:
    """forward of model on device"""
    graph_def, state = mpd.device(device)(nnx.split)(model)
    return state

def forward(sl_model: SLModel, x: Dict[str, MPObject]) -> MPObject:
    """forward of SLModel"""
    if x.keys() != sl_model.base_models.keys():
        raise ValueError("x keys must be equal to base_models keys")
    for k, v in x.items():
        sl_model.base_models[k].forward(v)

@mpd.function
def temp(graph_def, state, x):
    graph_def_d = mpd.device("P0")(lambda x:x)(graph_def)
    state_d = mpd.device("P0")(lambda x:x)(state)
    model = mpd.device("P0")(nnx.merge)(graph_def_d, state_d)
    # 3. Call the `nnx.Module`
    y = mpd.device("P0")(model)(x)
    # 4. Use `nnx.split` to propagate `nnx.State` updates.
    _, state = mpd.device("P0")(nnx.split)(model)
    return y, state

@jax.jit
def temp_jax(graph_def, state, x):
    model = (nnx.merge)(graph_def, state)
    # 3. Call the `nnx.Module`
    y = model(x)
    # 4. Use `nnx.split` to propagate `nnx.State` updates.
    _, state = nnx.split(model)
    return y, state

if __name__ == "__main__":
    from sfl_lite.ml.nn.models import ModelFactory
    import mplang as mp
    rngs = nnx.Rngs(42)
    cluster_spec = mp.ClusterSpec.from_dict({
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
    })
    sim = mp.Simulator(cluster_spec)
    
    rngs = nnx.Rngs(42)
    model = ModelFactory.create_model("dnn", input_dim=10, num_classes=3, rngs=rngs)
    graphdef, state = nnx.split(model)

    # Create sample data
    batch_size = 4
    x = jax.random.normal(rngs(), (batch_size, 10))
    z = mp.evaluate(sim, temp, graphdef, state, x)
    
    print("z:", z, mpd.fetch(sim, z))
    # z_jax = temp_jax(graphdef, state, x)
    # print("z_jax:", z_jax)
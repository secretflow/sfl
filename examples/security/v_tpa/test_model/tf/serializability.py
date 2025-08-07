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

import pdb
import pickle

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def unpack(model, training_config, model_config, weights):
    custom_objects = model_config.get("custom_objects", None)
    restored_model = deserialize(model, custom_objects=custom_objects)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(training_config)
        )
    restored_model.set_weights(weights)
    return restored_model


# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("get_config", None)
        model_config = model_metadata.get("model_config", None)
        model_config["custom_objects"] = {model_config["class_name"]: type(self)}
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, model_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


# Run the function
# make_keras_picklable()

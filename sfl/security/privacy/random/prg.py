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

from abc import ABC, abstractmethod

class PRGInterface(ABC):
    """Interface for Pseudo-Random Generator (PRG) implementations."""

    @abstractmethod
    def generate(self, length: int) -> bytes:
        """Generate a pseudo-random byte string of the specified length.

        Args:
            length (int): The length of the byte string to generate.

        Returns:
            bytes: A pseudo-random byte string of the specified length.
        """
        pass

    @abstractmethod
    def seed(self, seed_value: bytes) -> None:
        """Seed the PRG with a specific value.

        Args:
            seed_value (bytes): The seed value to initialize the PRG.
        """
        pass`
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

import abc
from typing import Any, Dict, Optional

from mplang.v1.core import MPObject


class LinearRegressionTemplate(abc.ABC):
    """
    Template abstract class for federated linear models with sklearn-like interface.

    This template provides a foundation for implementing federated linear regression
    models that are compatible with scikit-learn's interface while supporting
    secure multi-party computation (MPC).
    """

    @abc.abstractmethod
    def fit(
        self,
        X: Dict[str, MPObject],
        y: MPObject,
        sample_weight: Optional[MPObject] = None,
        **kwargs,
    ) -> "LinearRegressionTemplate":
        """
        Fit the linear model to training data.

        Args:
            X: Dict mapping party names to their feature data as MPObjects
            y: Target values as MPObject
            sample_weight: Optional sample weights as MPObject
            **kwargs: Additional keyword arguments (e.g., label_party, world_size)

        Returns:
            self: Fitted model instance
        """
        pass

    @abc.abstractmethod
    def predict(self, X: Dict[str, MPObject], **kwargs) -> MPObject:
        """
        Make predictions using the fitted model.

        Args:
            X: Dict mapping party names to their feature data as MPObjects
            **kwargs: Additional keyword arguments

        Returns:
            y_pred: Predicted values as MPObject
        """
        pass

    @abc.abstractmethod
    def score(
        self,
        X: Dict[str, MPObject],
        y: MPObject,
        sample_weight: Optional[MPObject] = None,
        **kwargs,
    ) -> MPObject:
        """
        Return the coefficient of determination R^2 of the prediction.

        Args:
            X: Dict mapping party names to their feature data as MPObjects
            y: True values as MPObject
            sample_weight: Optional sample weights as MPObject
            **kwargs: Additional keyword arguments

        Returns:
            score: R^2 score as MPObject
        """
        pass

    @abc.abstractmethod
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep: If True, return parameters for this estimator and
                 contained subobjects that are estimators

        Returns:
            params: Parameter names mapped to their values
        """
        pass

    @abc.abstractmethod
    def set_params(self, **params) -> "LinearRegressionTemplate":
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters

        Returns:
            self: Estimator instance
        """
        pass

    # Properties that sklearn estimators typically have
    @property
    @abc.abstractmethod
    def coef_(self) -> Dict[str, MPObject]:
        """Coefficients of the linear model distributed across parties."""
        pass

    @property
    @abc.abstractmethod
    def intercept_(self) -> Optional[MPObject]:
        """Intercept (bias) term of the linear model."""
        pass

    @property
    @abc.abstractmethod
    def n_features_in_(self) -> int:
        """Number of features seen during fit."""
        pass

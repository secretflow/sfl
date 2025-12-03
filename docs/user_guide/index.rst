.. _user_guide:

User Guide
============

Federated Learning
------------------

Federated learning is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them.


.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Horizontal Federated Learning
      :link: federated_learning/horizontal_federated_learning/index
      :link-type: doc

        For cases that multi participants share the same feature space but differ in sample ID.

    .. grid-item-card:: Vertical Federated Learning
      :link: federated_learning/vertical_federated_learning/index
      :link-type: doc

        For cases that multi participants share the same sample ID space but differ in feature space.

    .. grid-item-card:: Mix Federated Learning
      :link: federated_learning/mix_federated_learning
      :link-type: doc

        For cases that parts of participants share the same sample ID space but differ in feature space,
        where others share the same feature space but differ in sample ID.

.. toctree::
   :hidden:
   :maxdepth: 2

   federated_learning/index


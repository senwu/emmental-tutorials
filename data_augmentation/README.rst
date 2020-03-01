Uncertainty-based sampling data augmentation
============================================

In this advanced tutorial, we study how to make data augmentation more efficient. We propose an uncertainty-based random sampling scheme which, among the transformed data points, picks those with the highest losses, i.e. those "providing the most information". We will build an Emmental_ application to tackle the image classification benchmarks (i.e. MNIST, CIFAR-10, and CIFAR-100) to better understand data augmentation.

Installation
------------

To use this tutorial you will need to:

1. Install the package, Emmental data augmentation (EDA), and any other Python dependencies by running:

.. code:: bash

  make dev

2. Run the image classification task with the following command. For more details, please check `run.sh`

.. code:: bash

  bash run.sh \
         ${TASK} \
         ${SEED} \
         ${GPU_IDS} \
         ${MODEL_NAME} \
         uncertainty_sampling \
         ${BATCH_SIZE} \
         ${NUM_COMP} \
         ${AUGMENT_K} \
         ${AUGMENT_ENLARGE}

Acknowledgements
----------------

Much of the code in this tutorial was adapted from the `Fast AutoAugment`_. We thank all authors to provide these available online.

.. _Emmental: https://github.com/SenWu/emmental
.. _Fast AutoAugment: https://github.com/kakaobrain/fast-autoaugment

Uncertainty-based sampling data augmentation
============================================

In this advanced tutorial, we study how to make data augmentation more efficient. We propose an uncertainty-based random sampling scheme which, among the transformed data points, picks those with the highest losses, i.e. those "providing the most information". We will build an Emmental_ application to tackle the image classification benchmarks (i.e. MNIST, CIFAR-10, and CIFAR-100) to better understand data augmentation.

Installation
------------

To use this tutorial you will need to:

1. Install the package, Emmental data augmentation (EDA), and any other Python dependencies by running:

.. code:: bash

  make dev

2. Run the image classification task with the following command.

.. code:: bash

  bash run.sh \
         ${TASK} \
         ${SEED} \
         ${GPU_IDS} \
         ${MODEL_NAME} \
         ${AUGMENT_POLICY} \
         ${BATCH_SIZE} \
         ${NUM_COMP} \
         ${AUGMENT_K} \
         ${AUGMENT_ENLARGE}

The default `AUGMENT_POLICY` policy method is `uncertainty_sampling` which concatenates composition of ${NUM_COMP} random selected transformations and default transformations. For more details, please check `run.sh`.

Specify your augmentation [Optional]
----------------------------------------

We provide 16 transformations in the tutorial:

.. code:: bash

  AutoContrast
  Brightness
  Color
  Contrast
  Cutout
  Equalize
  Invert
  Mixup
  Posterize
  Rotate
  Sharpness
  ShearX
  ShearY
  Solarize
  TranslateX
  TranslateY

For each transformation, you can set the probability and magnitude of applying the transformation (i.e. `AutoContrast_P{PROBABILITY}_L{MAGNITUDE}`), otherwise they are all random. You can also composite different transformation by concatenating them with `@` (i.e. `AutoContrast@Color`).

Acknowledgements
----------------

Much of the code in this tutorial was adapted from the `Fast AutoAugment`_. We thank all authors to provide these available online.

.. _Emmental: https://github.com/SenWu/emmental
.. _Fast AutoAugment: https://github.com/kakaobrain/fast-autoaugment

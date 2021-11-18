Uncertainty-based sampling data augmentation
============================================

In this advanced tutorial, we study how to make data augmentation more efficient. We propose an uncertainty-based random sampling scheme which, among the transformed data points, picks those with the highest losses, i.e. those ``providing the most information``. We will build an Emmental_ application to tackle the image classification benchmarks (i.e. MNIST, CIFAR-10, and CIFAR-100) to better understand data augmentation.

Installation
------------

To use this tutorial you will need to install the package, Emmental data augmentation (EDA), and any other Python dependencies by running:

.. code:: bash

  make dev

Running End-to-end image classification task
--------------------------------------------

To run image classification, we provide a simple `run.sh` and you just run the following command.

.. code:: bash

  bash run.sh

The default ``augment_policy`` is ``uncertainty_sampling`` which concatenates composition of 2 randomly selected transformations and default transformations (i.e. randomly cropping, horizontal flipping, cutout and mixup). We also provide a command-line interface for each parameter. For more detailed options, run ``image -h`` to see a list of all possible options.

Specify your augmentation [Optional]
----------------------------------------

We provide several transformations in the tutorial, here are some examples:

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

For each transformation, you can set the probability and magnitude of applying the transformation (i.e. ``AutoContrast_P{PROBABILITY}_L{MAGNITUDE}``), otherwise they are all random. You can also composite different transformation by concatenating them with ``@`` (i.e. ``AutoContrast@Color``).

Acknowledgments
---------------

Much of the code in this tutorial was adapted from the `Fast AutoAugment`_. We thank all authors to provide these available online.

.. _Emmental: https://github.com/senwu/emmental
.. _Fast AutoAugment: https://github.com/kakaobrain/fast-autoaugment

CheXNet
=======

In this advanced tutorial, we will build an Emmental_ application to tackle the
CheXNet X-ray classification problem which is to predict 14 common diagnoses in
over 100,000 NIH chest X-rays proposed from the `CheXNet paper`_.

Installation
------------

To use this tutorial you will need to:

1. Install python dependencies for this tutorial.

.. code:: bash

  pip install -r requirements.txt

2. Download the CheXNet dataset (including all images and meta information file
(e.g. nih_labels.csv)) used in the tutorial to the local directory.

3. Run the CheXNet tasks with the following command, where ``TASK`` is the task name
list delimited by ``,``, any combination from ``Atelectasis``, ``Cardiomegaly``,
``Effusion``, ``Infiltration``, ``Mass``, ``Nodule``, ``Pneumonia``, ``Pneumothorax``,
``Consolidation``, ``Edema``, ``Emphysema``, ``Fibrosis``, ``Pleural_Thickening``, and
``Hernia``.

.. code:: bash 

  bash run.sh [DATAPATH] [IMAGEPATH] [TASK] [LOG] [SEED] [GPU] [LR]

Acknowledgments
---------------

Much of the code in this tutorial was adapted from the reproduce-chexnet_. We thank
all authors to provide these available online.

.. _Emmental: https://github.com/SenWu/emmental
.. _reproduce-chexnet: https://github.com/jrzech/reproduce-chexnet
.. _`CheXNet paper`: https://arxiv.org/pdf/1711.05225

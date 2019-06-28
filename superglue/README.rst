SuperGLUE
=========

In this advanced tutorial, we will build an Emmental_ application to tackle the
SuperGLUE, a new benchmark styled after GLUE with a new set of more difficult
language understanding tasks.

Installation
------------

To use this tutorial you will need to:

1. Install python dependencies for this tutorial.

.. code:: bash

  pip install -r requirements.txt

2. Download the SuperGLUE and other datasets (e.g., SWAG_) used in the tutorial to the local directory.

.. code:: bash
  bash download_data.sh [DATA]

3. Run the SuperGLUE task with the following command, where `TASK` is one of `cb`, `copa`, `multirc`, `rte`, `wic`, `wsc` and external task for pretraining (e.g., `swag`).

.. code:: bash 

  bash run_superglue.sh [TASK] [DATA] [SEED] [GPU_ID]

Pretraining [Optional]
----------------------
We use pretrained model to improve model perfromance, for example we use `MNLI` model from `GLUE tutorial <../glue/>`_ for `RTE` and `CB` task, and use `SWAG` model to improve `COPA` task. To get the pretrained model, you can train `MNLI` model using GLUE tutorial and `SWAG` using this tutorial.


**Note:** Due to small validation sets and evidence of overfitting to the validation set on some of the tasks (CB, COPA, and WiC), we also recommend using cross-validation on these tasks. For each, we ran k-fold cross-validation with a value of k that would result in a validation set of approximately the same size as the original provided split.

Acknowledgements
----------------

Much of the code in this tutorial was adapted from the jiant_. We thank all authors to provide these available online.

.. _Emmental: https://github.com/SenWu/emmental
.. _jiant: https://github.com/jsalt18-sentence-repl/jiant
.. _SWAG: https://github.com/rowanz/swagaf


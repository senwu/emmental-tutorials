General Language Understanding Evaluation (GLUE)
================================================

In this advanced tutorial, we will build an Emmental_ application to tackle the
General Language Understanding Evaluation (GLUE) benchmark which is a collection
of resources for training, evaluating, and analyzing natural language understanding
systems.

Installation
------------

To use this tutorial you will need to:

1. Download the GLUE data to the local directory.

.. code:: python

  python download_glue_data.py -d [GLUEDATA]

2. Run the GLUE task with the following command, where `TASK` is task name list delimited by ",", any combination from `CoLA`, `MNLI`, `MRPC`, `QNLI`, `QQP`, `RTE`, `SST-2`, `STS-B`, `WNLI`.

.. code:: bash

  bash run_glue.sh ${TASK} ${GLUEDATA} ${SEED} ${GPU_ID}

Acknowledgments
---------------


Much of the code in this tutorial was adapted from the jiant_. We thank all authors to provide these available online.

.. _Emmental: https://github.com/senwu/emmental
.. _jiant: https://github.com/jsalt18-sentence-repl/jiant


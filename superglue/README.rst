SuperGLUE
=========

In this advanced tutorial, we will build a Emmental_ application to tackle the
SuperGLUE, a new benchmark styled after GLUE with a new set of more difficult
language understanding tasks, improved resources.

Installation
------------

To use this tutorial you will need to:

  1. Install python dependencies for this tutorial.

  .. code:: bash

    pip install -r requirements.txt

  2. Download the SuperGLUE data to local director.

  .. code:: python

    python download_superglue_data.py --data_dir [SUPERGLUEDATA]

  3. Run the SuperGLUE tasks using following command, where `TASK` is one of
    `cb`, `copa`, `multirc`, `rte`, `wic`, `wsc`.

  .. code:: bash 

    bash run_superglue.sh [TASK] [SUPERGLUEDATA] [SEED] [GPU_ID]


.. _Emmental: https://github.com/SenWu/emmental


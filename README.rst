Emmental Tutorials
===================

Introduction Tutorials
----------------------

We have several `introductory tutorials <intro/>`_ to help get you started with using Emmental.

Chexnet
-------

In this `tutorial <chexnet/>`_, we build an Emmental application to predicted 14 common diagnoses using convolutional neural networks in over 100,000 NIH chest x-rays proposed from the `CheXNet paper`_.

GLUE Tutorial
-------------

In this `tutorial <glue/>`_, we build an Emmental application to tackle the General Language Understanding Evaluation (GLUE) benchmark which is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

SuperGLUE Tutorial
-------------

In this `tutorial <superglue/>`_, we build an Emmental application to tackle the SuperGLUE, a new benchmark styled after GLUE with a new set of more difficult language understanding tasks, improved resources, which achieves the `start-of-the-art score`_ on SuperGLUE benchmark on June 15, 2019 undert the name of Stanford Hazy Research.


Installation
------------

For the Python dependencies, we recommend using a
`virtualenv`_. Once you have cloned the
repository, change directories to the root of the repository and run

.. code:: bash

    virtualenv -p python3 .venv


Once the virtual environment is created, activate it by running:

.. code:: bash

    source .venv/bin/activate


Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run:

.. code:: bash

    deactivate


Then, install Fonduer and any other python dependencies by running:

.. code:: bash

    pip install -r requirements.txt

Running
~~~~~~~

After installing all the requirements, just run:

.. code:: bash

    jupyter notebook


.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _`CheXNet paper`: https://arxiv.org/pdf/1711.05225
.. _`start-of-the-art score`: https://super.gluebenchmark.com/leaderboard

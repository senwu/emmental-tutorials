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

In this `tutorial <superglue/>`_, we build an Emmental application to tackle `SuperGLUE`_, a new benchmark in the same style as GLUE with a set of more difficult language understanding tasks. Our submission achieved a new `start-of-the-art score`_ on June 15, 2019 under the name of Stanford Hazy Research. This code has also been refactored and used in the `Snorkel`_ project in the `snorkel-superglue`_ repository.


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


Then, install Emmental and any other python dependencies by running:

.. code:: bash

    pip install -r requirements.txt


.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _`CheXNet paper`: https://arxiv.org/pdf/1711.05225
.. _`start-of-the-art score`: https://super.gluebenchmark.com/leaderboard
.. _`SuperGLUE`: https://super.gluebenchmark.com
.. _`Snorkel`: http://snorkel.stanford.edu
.. _`snorkel-superglue`: https://github.com/HazyResearch/snorkel-superglue

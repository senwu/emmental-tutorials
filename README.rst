Emmental Tutorials
===================

Introduction Tutorials
----------------------

We have several `introductory tutorials <intro/>`_ to help get you started with using Emmental.

GLUE Tutorial
-------------

In this `tutorial <glue/>`_, we build an Emmental application to tackle the General Language Understanding Evaluation (GLUE) benchmark which is a collection of resources for training, evaluating, and analyzing natural language understanding systems.


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
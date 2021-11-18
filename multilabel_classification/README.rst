Multilabel classification
=========================

In this advanced tutorial, we will build a multilabel classification application using
Emmental_. We choose `Toxic Comment Classification Challenge`_ from Kaggle which aims
to identify and classify toxic online comments into six categories: toxic,
severe_toxic, obscene, threat, insult, and identity_hate.

Installation
------------

To use this tutorial you will need to:

1. Install python dependencies for this tutorial.

.. code:: bash

  pip install -r requirements.txt

2. Download the dataset from Kaggle and split your data into train, val, and test
split if needed.

3. Run the multilabel classification with the following command, and you can see the
details in the script:

.. code:: bash 

  bash run.sh

Acknowledgments
---------------

We thank all authors to provide dataset available online.

.. _Emmental: https://github.com/senwu/emmental
.. _`Toxic Comment Classification Challenge`: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

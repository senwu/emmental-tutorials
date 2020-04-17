Text Classification
================================================

In this advanced tutorial, we will build an Emmental_ application to tackle the text
classification. More specifically, we analyze how information transfer among different
tasks in mulit-task learning settings.

Running End-to-end text classification task
--------------------------------------------

1. Download the text classification dataset used in the tutorial to the local directory
[DATA] using the following command.

.. code:: bash

  bash download_data.sh [DATA]

2. Download pre-trained word embeddings such as glove_ and make it into text format.

.. code:: bash

  wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
  unzip glove.6B.zip

3. Run the text classification task with the following command, where ``TASK`` is the
task name list delimited by ``,``, any combination from ``mr``, ``sst``, ``subj``,
``cr``, ``mpqa``, ``trec``; ``MODEL`` is one of ``lstm``, ``cnn``, or ``mlp``. More
details can be found in script.

.. code:: bash

  bash run_text_classification.sh ${TASK} [DATA] [MODEL] [LOGPATH] [SEED] [GPU_ID] [EMB]

Acknowledgements
----------------

Much of the code in this tutorial was adapted from the sru_. We thank all authors to provide these available online.

.. _Emmental: https://github.com/SenWu/emmental
.. _sru: https://github.com/taolei87/sru
.. _glove: http://nlp.stanford.edu/projects/glove/


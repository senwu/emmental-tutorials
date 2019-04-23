# Emmental Tutorials

## Introduction Tutorials

We have several [introductory tutorials](intro/) to help get you started with using Emmental.

## GLUE Tutorial

In this [tutorial](GLUE/), we build an Emmental application to tackle the General Language Understanding Evaluation (GLUE) benchmark which is a collection of resources for training, evaluating, and analyzing natural language understanding systems.


## Installation

For the Python dependencies, we recommend using a
[virtualenv](https://virtualenv.pypa.io/en/stable/). Once you have cloned the
repository, change directories to the root of the repository and run

```bash
virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```bash
source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run `deactivate`.

Then, install Fonduer and any other python dependencies by running:

```bash
pip install -r requirements.txt
```

### Running

After installing all the requirements, just run:

```
jupyter notebook
```
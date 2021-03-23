# Ought Text Classification
> Few-shot text classification on scientific abstracts


## Installation

The `ought` package is currently not being distributed. But it can be installed locally with git:

```
git clone https://github.com/iyaja/ought.git
cd ought
pip install .
```

## Usage

The top-level `ought` module is plit into 5 main submodules:

1. `starter`: The provided starter code, which contains a suite of utility functions for GPT-2
2. `gpt`: Uses GPT-2 in multiple ways to expose 3 different classifiers.
2. `lstm`: Implements LSTM models that train on a few examples at test time.
2. `bart`: Implements BART-based classifier.
2. `metrics`: Final metrics/scores for all models and ensemble classifier.

All classification models provided are of the form `XXClassifier` where `XX` is the model name. Currently, the implemented classifiers are:

- `GPTLMClassifier`: A GPT-2 language-model based classifier.
- `GPTMatmulClassifier`: A few-shot shot GPT-2 classifier that computes prediction scores though matrix multiplication of stacked hidden states.
- `GPTSimilarityClassifier`: A few-shot shot GPT-2 classifier that computes prediction scores though dot products of hidden states.
- `LSTMClassifier`: A classifier that requires few-shot training using an LSTM. Training can be completed in under a minute for 500+ samples
- `BARTClassifier`: A zero-shot classifier based on the BART architecure.

All of the above models confirm to a standard interface. They all have an initializer that takes in a path to a `jsonl` file with context examples and a `samples` parameter to control how many xontextual samples to use per label/class. Once initialized, they all expose a `predict` method that takes in a string and returns either `"AI"` or `"Not AI`. The results and accuracy for each model can be found in the `04_metrics.ipynb` notebook or, equaivalently, in the [documentation site](https://iyaja.github.io/ought/metrics.html).

Here is an example of using the `GPTMatmulClassifier`:

```python
from ought.gpt import GPTMatmulClassifier
model = GPTMatmulClassifier(json='data/train.jsonl', samples=5)
model.predict("In this paper, we propose a new zero-shot transformer-based algorithm to classifiy scientific papers as AI or NOT AI.")
```




    'AI'



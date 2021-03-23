# AUTOGENERATED! DO NOT EDIT! File to edit: 03_lstm.ipynb (unless otherwise specified).

__all__ = ['LSTMClassifier']

# Cell
from .starter import *
import fastai
from fastai.text.all import *

# Cell
class LSTMClassifier:
    def __init__(self, json='data/train.jsonl', samples=5, show_metrics=[]):
        self.path = train_json_path
        self.df = pd.DataFrame(uniform_samples(json, samples))
        self.dls = TextDataLoaders.from_df(self.df, path=train_json_path, text_col='text', label_col='label', valid_col=None, seq_len=50)
        self.learn = text_classifier_learner(self.dls, AWD_LSTM, drop_mult=0.5, metrics=metrics)
        self.learn.fine_tune(5, 5e-2)

    def predict(self, prompt):
        pred = self.learn.predict(prompt)[0]
        return 'NOT AI' if (pred == 'False') else 'AI'
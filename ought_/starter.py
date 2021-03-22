# AUTOGENERATED! DO NOT EDIT! File to edit: starter.ipynb (unless otherwise specified).

__all__ = ['generate', 'load_jsonl', 'render_example', 'Label', 'render_end_example', 'make_prompt']

# Cell
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

# Cell
def generate(prompt, max_length=5, stop_token=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_text_ids = model.generate(input_ids=input_ids.cuda(), max_length=max_length+len(input_ids[0]), do_sample=False)
    generated_text = tokenizer.decode(generated_text_ids[0], clean_up_tokenization_spaces=True)
    post_prompt_text = generated_text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)):]
    return prompt + post_prompt_text[:post_prompt_text.find(stop_token) if stop_token else None]

# Cell
def load_jsonl(filename):
    f = open(filename)
    return [json.loads(line) for line in f.read().splitlines()]

# Cell
def render_example(example):
    title = example["text"].split(".")[0].strip()
    abstract = example["text"][len(title)+1:].strip()
    return f"""Title: {title}
Abstract: {abstract}
Label: {"AI" if example["label"] == "True" else "Not AI"}"""

# Cell
def render_end_example(example):
    title = example["text"].split(".")[0].strip()
    abstract = example["text"][len(title)+1:].strip()
    return f"""Title: {title}
Abstract: {abstract}
Label:"""

# Cell
def make_prompt(instructions, train_examples, end_example):
    rendered_train_examples = "\n\n--\n\n".join([render_example(example) for example in train_examples])
    return f"""{instructions}

{rendered_train_examples}

--

{render_end_example(end_example)}"""
# AUTOGENERATED! DO NOT EDIT! File to edit: 01_starter.ipynb (unless otherwise specified).

__all__ = ['tokenizer', 'model', 'generate', 'load_jsonl', 'render_example', 'render_end_example', 'make_prompt',
           'GPTClassifier']

# Cell
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

# Cell
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval().cuda()

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
    return f'TITLE: {title}\nABSTRACT: {abstract}\nLABEL: {"AI" if example["label"] == "True" else "NOT AI"}'

# Cell
def render_end_example(example):
    title = example["text"].split(".")[0].strip()
    abstract = example["text"][len(title)+1:].strip()
    return f"TITLE: {title}\nABSTRACT: {abstract}\nLABEL:"

# Cell
def make_prompt(instructions, train_examples, end_example):
    rendered_train_examples = "\n\n--\n\n".join([render_example(example) for example in train_examples])
    return f"""{instructions}

{rendered_train_examples}

--

{render_end_example(end_example)}"""

# Cell
class GPTClassifier:
    def __init__(self, instructions='Label each of the following examples as "AI" or "NOT AI"', json='data/train.jsonl', samples=4):
        self.instructions = instructions
        self.context = load_jsonl(json)[:samples]

    def predict(self, prompt):
        prompt = make_prompt(self.instructions, self.context, {'text': prompt})
        out = generate(prompt, stop_token="\n")
        # to create a concrete prediction, take the last line and strip the "LABEL: " component
        pred = out.split('\n')[-1].strip("LABEL: ")
        return pred
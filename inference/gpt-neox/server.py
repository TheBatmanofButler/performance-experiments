from flask import Flask
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

import infer

app = Flask(__name__)

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

@app.route("/<prompt>")
def run_inference(prompt):
    return infer.infer(prompt, model, tokenizer)

app.run(host='0.0.0.0', port=81)

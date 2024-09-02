
def infer(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )

    return tokenizer.batch_decode(gen_tokens)[0]

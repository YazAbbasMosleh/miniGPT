from verdict.preprocessV1 import preprocessed

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# print(f"all-tokens: {all_tokens}")
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(f"vocab type: {type(vocab)}")
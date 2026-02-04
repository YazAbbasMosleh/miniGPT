from verdict.preprocessV1 import preprocessed

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(f"Vocabulary size: {vocab_size}")


vocab = {token:integer for integer, token in enumerate(all_words)}
print(f"vocab type: {type(vocab)}")
for i, item in enumerate(vocab.items()):
    print(f"Index: {i}, Token: {item[0]}, Integer: {item[1]}")
    if i >= 50:
        break
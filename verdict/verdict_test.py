with open("verdict/the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
import re

from verdict.verdict_test import raw_text

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Len preprocessed: {len(preprocessed)}")



# print(f"*******type of preprocessed: {type(preprocessed)}")
with open("preprocessed.txt", "w") as f:
    for item in preprocessed:
        f.write(f"{item}\n")
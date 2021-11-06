import os
import json
import re
import tqdm

def clean_data(idx, data):
    text = data["text"]
    text = re.sub('\|', ' ', text)
    text = re.sub('p=[0-9 ]*', '', text)
    text = re.sub('링크=[\S]*', '', text)
    text = re.sub('http\S+', '', text)
    text = re.sub('[0-9가-힣a-zA-Z]*=[0-9가-힣a-zA-Z–§]*', '', text)
    text = re.sub('[;#*]', '', text)
    text = re.sub('\\\\n', '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\s+', ' ', text)

    data["text"] = text

    return data

def append_title(idx, data):
    text = data["text"]
    title = data["title"]

    text = re.sub('#', '', text)

    data["text"] = f'#{title}# {text}'

    return data

def main():
    data_path = "../data"
    context_path = "wikipedia_documents.json"
    new_context_path = "wikipedia_documents_cleaned.json"

    with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    new_wiki = {}
    for idx in wiki:
        cleaned_data = clean_data(idx, wiki[idx])
        new_wiki[idx] = cleaned_data

    with open(os.path.join(data_path, new_context_path), "w+", encoding="utf-8") as f:
        json.dump(new_wiki, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
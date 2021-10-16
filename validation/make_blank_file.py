import ast
import re
import json

dictionary = {}

with open('test_dataset.txt', 'r', encoding='utf-8') as f:
    contents = f.read()

contents_after = re.sub('}{', '}\n{', contents)
contents_after = contents_after.split("\n")

with open('validation_test.json', 'w+', encoding='utf-8') as f:
    f.write("{"+"\n")
    for idx, line in enumerate(contents_after):
        res = ast.literal_eval(line)
        dictionary[res['id']] = ""

        if idx != len(contents_after) -1 :
            f.write("    \""+res['id']+"\"" + ':' + '"",' + "\n")
        else:
            f.write("    \""+res['id']+"\"" + ':' + '""' + "\n")

    f.write("}"+"\n")
import json
preds = json.load(open("results_squad2/predictions.json", encoding="utf-8"))
print("GT:", len(gt))           # load_squad
print("len:", len(preds))      # predictions.json

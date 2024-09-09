from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from sklearn.metrics import r2_score
import string
import csv

# Download and load XCOMET model
# model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model_path = download_model("Unbabel/XCOMET-XL")
# model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
model = load_from_checkpoint(model_path)

# Define file paths
CSV_FILEPATH = "annotations/port_bi_nllb_lora.csv"
OUTPUT_FILEPATH = "output-xcomet_bi_nllb_lora_no_commas_0.csv"

items = []

with open(CSV_FILEPATH, encoding="utf-8", mode="r") as fr:
    csvreader = csv.reader(fr, delimiter=',')
    for item in csvreader:
        try:
            src = item[0][0] + item[0][1:].lower()
            ref = item[1].translate(str.maketrans('', '', string.punctuation))
            mt = item[2].translate(str.maketrans('', '', string.punctuation))
            items.append({
                "src": src,
                "ref": ref,
                "mt": mt
            })
            # print({
            #     "src": src,
            #     "ref": ref,
            #     "mt": mt
            # })
        except Exception as e:
            print(e)
            print("Error:", item)

# # Read source, reference, and predicted files
# with open(SRC_FILE, 'r') as src:
#     srcs = [line.rstrip() for line in src]

# with open(REF_FILE, 'r') as ref:
#     refs = [line.rstrip() for line in ref]

# with open(PRED_FILE, 'r') as pred:
#     preds = [line.rstrip() for line in pred]

metric = BLEU(
        lowercase=False, tokenize='spm', force=False,
        smooth_method='exp', smooth_value=None,
        effective_order=True)

# Evaluate and write results to CSV file
with open(OUTPUT_FILEPATH, 'w') as output:
    print("Num_items:", len(items))
    # Calculate reference-free score using XCOMET model
    ref_free_scores = model.predict(items, batch_size=4).scores

    # Calculate BLEU score
    # bleu_score = [metric.sentence_score(preds[idx], [ref_text]).score/100 for idx, ref_text in enumerate(tqdm(refs))]
    bleu_score = [metric.sentence_score(item["mt"].lower(), [item["ref"].lower()]).score/100 for item in items]

    for bleu, ref_free, item in zip(bleu_score, ref_free_scores, items):
        # Write results to CSV
        output_line = f"{item['src']}, {item['ref']}, {item['mt']}, {bleu}, {ref_free}\n"
        output.write(output_line)
        # print(output_line)

print(r2_score(ref_free_scores, bleu_score))
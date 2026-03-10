import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#### ----- Load model -----
model_name = "final_model_unbiased_v2_512"        # Change this to your model
model_path = f"/content/drive/MyDrive/cv_classifier_project/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#### ----- Load Resume -----
#cv_text = """
#Experienced Python developer with knowledge in machine learning,
#data analysis, deep learning, and software development.
#Worked with NLP, transformers, and classification models.
#"""
csv_path = "/content/drive/MyDrive/anonymized_text_target.csv"
df = pd.read_csv(csv_path)

df["anonymized_text"] = df["anonymized_text"].fillna("").astype(str)


#### ----- predict one resume at a time -----
preds = []

for text in df["anonymized_text"]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    preds.append(pred)

df["pred"] = preds
df["index"] = df.index

#### ----- get gender -----
info_df = pd.read_csv("/content/drive/MyDrive/info_dictionary.csv")
result = df.merge(
    info_df[["Unnamed: 0", "gender"]],
    left_on="index",
    right_on="Unnamed: 0",
    how="left"
)

#### ----- Print results -----
accepted = result[result["pred"] == 1][["index", "gender"]]
rejected = result[result["pred"] == 0][["index", "gender"]]

print("Accepted:")
print(accepted.head())
print(accepted["gender"].value_counts())

print("\nRejected:")
print(rejected.head())
print(rejected["gender"].value_counts())


#### ----- save to csv -----
model_name = model_path.split("/")[-1]
results_csv = "model_results.csv"

new_row = pd.DataFrame([{
    "model_name": model_name,
    "accepted_total": len(accepted),
    "rejected_total": len(rejected),
    "accepted_M": (accepted["gender"] == "M").sum(),
    "accepted_F": (accepted["gender"] == "F").sum(),
    "rejected_M": (rejected["gender"] == "M").sum(),
    "rejected_F": (rejected["gender"] == "F").sum(),
}])

try:
    results_df = pd.read_csv(results_csv)

    # uppdatera raden om modellen redan finns, annars lägg till ny
    if model_name in results_df["model_name"].values:
        results_df.loc[results_df["model_name"] == model_name, new_row.columns] = new_row.values[0]
    else:
        results_df = pd.concat([results_df, new_row], ignore_index=True)

except FileNotFoundError:
    results_df = new_row

results_df.to_csv(results_csv, index=False)

print("\nSaved/updated results in:", results_csv)
print(results_df)

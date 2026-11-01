import os
import numpy as np, pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

LABELS = ["Credit Risk", "Liquidity Risk", "Operational Risk", "Market Growth"]
l2i = {l: i for i, l in enumerate(LABELS)}
i2l = {i: l for l, i in l2i.items()}

df = pd.read_csv("data/risk_labeled.csv").dropna(subset=["text", "label"])
df = df[df.label.isin(LABELS)].copy()
df["label_id"] = df.label.map(l2i)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label_id, random_state=42)
train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
print("Train:", len(train_df), "Test:", len(test_df))

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def to_ds(d):
    ds = Dataset.from_pandas(d[["text", "label_id"]].rename(columns={"label_id": "label"}),
                             preserve_index=False)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train_ds, test_ds = to_ds(train_df), to_ds(test_df)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4, id2label=i2l, label2id=l2i)

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1_macro": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(output_dir="models/ckpt", num_train_epochs=4,
                         per_device_train_batch_size=16, per_device_eval_batch_size=32,
                         learning_rate=3e-5, eval_strategy="epoch", save_strategy="no",
                         logging_steps=10, report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

# ---- Evaluation ----
out = trainer.predict(test_ds)
y_true, y_pred = out.label_ids, np.argmax(out.predictions, axis=1)
report = classification_report(y_true, y_pred, target_names=LABELS)
cm = confusion_matrix(y_true, y_pred)
print(report); print(cm)

# Accuracy per source filing (checks for "bank vs tech vocabulary" shortcut)
test_df["pred"] = y_pred
per_source = test_df.groupby("source").apply(
    lambda g: accuracy_score(g.label_id, g.pred), include_groups=False)
print("Accuracy by source:\n", per_source)

# Optional hand-written test set
hand_msg = ""
if os.path.exists("eval/hand_test.csv"):
    h = pd.read_csv("eval/hand_test.csv").dropna()
    h = h[h.label.isin(LABELS)]
    h_ds = to_ds(h.assign(label_id=h.label.map(l2i)))
    hp = np.argmax(trainer.predict(h_ds).predictions, axis=1)
    hand_msg = f"Hand-written test accuracy: {accuracy_score(h.label.map(l2i), hp):.2f} ({len(h)} sentences)"
    print(hand_msg)

os.makedirs("eval", exist_ok=True)
with open("eval/classifier_report.txt", "w", encoding="utf-8") as f:
    f.write(report + "\nConfusion matrix:\n" + str(cm) +
            "\n\nAccuracy by source:\n" + str(per_source) + "\n" + hand_msg)

trainer.save_model("models/risk_classifier")
tok.save_pretrained("models/risk_classifier")
print("Saved model to models/risk_classifier")
import re, glob, pandas as pd

KEYWORDS = {
    "Credit Risk":      ["default", "credit loss", "counterparty", "delinquen", "allowance for", "borrower", "credit quality", "nonperforming"],
    "Liquidity Risk":   ["liquidity", "cash flow", "funding", "debt maturit", "credit facility", "working capital", "refinanc", "capital requirements"],
    "Operational Risk": ["supply chain", "cyber", "system failure", "fraud", "personnel", "disruption", "outage", "third-party", "data breach"],
    "Market Growth":    ["market share", "demand", "expansion", "growth", "new markets", "customers", "competitive", "competition"],
}
MAX_PER_CLASS = 150

def weak_label(s):
    low = s.lower()
    scores = {k: sum(w in low for w in v) for k, v in KEYWORDS.items()}
    top = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.get)
    # skip sentences with no match or a tie between classes
    if top[0] == 0 or top[0] == top[1]:
        return None
    return best

def looks_like_text(s):
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    return letters / max(len(s), 1) > 0.75 and digits / max(len(s), 1) < 0.1

rows = []
for fname in glob.glob("data/*.txt"):
    text = open(fname, encoding="utf-8").read()
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if 60 < len(sent) < 500 and looks_like_text(sent):
            lab = weak_label(sent)
            if lab:
                rows.append({"text": sent.strip(), "label": lab, "source": fname})

df = pd.DataFrame(rows).drop_duplicates("text")
df = (df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=42)))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/risk_labeled.csv", index=False)

print(df.label.value_counts())
print("Total:", len(df))
print(df.groupby(["source", "label"]).size())
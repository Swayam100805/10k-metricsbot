import re, glob, os
import fitz  # PyMuPDF

if not glob.glob("data/*.pdf"):
    raise SystemExit("No PDFs found in data/. Copy tech_10k.pdf and bank_10k.pdf there first.")

def clean(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)            # fix hyphenated line breaks
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)      # drop lone page numbers
    text = re.sub(r"\s*\n\s*", " ", text)                # join wrapped lines
    return re.sub(r"\s+", " ", text).strip()

for pdf in glob.glob("data/*.pdf"):
    doc = fitz.open(pdf)
    text = clean("\n".join(page.get_text() for page in doc))
    out = os.path.splitext(pdf)[0] + ".txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{out}: {len(doc)} pages, {len(text.split()):,} words")
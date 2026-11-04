import re, json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from retrieval import Retriever, numbers_in, first_number, is_numeric_question

QA_NAME = "distilbert-base-cased-distilled-squad"
GEN_NAME = "google/flan-t5-small"

qa_tok = AutoTokenizer.from_pretrained(QA_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_NAME).eval()
gen_tok = AutoTokenizer.from_pretrained(GEN_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_NAME).eval()

# ---------- Extractive: returns an exact span copied from the passage ----------
def snap_to_words(context, a, b):
    """Grow a span to whole words so '416' inside '416,161' becomes '416,161'."""
    while a > 0 and not context[a - 1].isspace():
        a -= 1
    while b < len(context) and not context[b].isspace():
        b += 1
    return context[a:b].rstrip(",;:.")

def extract_span(question, context, max_answer_tokens=30):
    enc = qa_tok(question, context, truncation="only_second", max_length=384,
                 return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    with torch.no_grad():
        out = qa_model(**enc)
    start_p = out.start_logits[0].softmax(-1).tolist()
    end_p = out.end_logits[0].softmax(-1).tolist()
    best_score, best_span = 0.0, None
    for i, s in enumerate(seq_ids):
        if s != 1:
            continue
        for j in range(i, min(i + max_answer_tokens, len(seq_ids))):
            if seq_ids[j] != 1:
                continue
            sc = start_p[i] * end_p[j]
            if sc > best_score:
                best_score, best_span = sc, (offsets[i][0], offsets[j][1])
    if best_span is None:
        return "", 0.0
    return snap_to_words(context, *best_span), best_score

def extractive_answer(question, retriever, k=5):
    """Run the reader on the top-k passages; combine reader confidence with retrieval rank."""
    numeric = is_numeric_question(question)
    best = {"answer": "", "score": 0.0, "context": ""}
    for hit in retriever.top(question, k):
        text, qa_score = extract_span(question, hit["text"])
        score = qa_score * (0.3 + 0.7 * hit["score"])
        if numeric and not re.search(r"\d", text):
            score *= 0.1                       # a money/ratio question needs a number
        if score > best["score"]:
            best = {"answer": text, "score": score, "context": hit["text"]}
    return best

def extractive_on_passage(question, passage):
    text, score = extract_span(question, passage)
    return {"answer": text, "score": score, "context": passage}

# ---------- Generative: FLAN-T5 writes a new answer ----------
def _generate(prompt, max_new_tokens, **gen_kwargs):
    enc = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        ids = gen_model.generate(**enc, max_new_tokens=max_new_tokens, **gen_kwargs)
    return gen_tok.decode(ids[0], skip_special_tokens=True)

def generative_answer(question, context):
    context = " ".join(context.split()[:300])
    return _generate(f"Answer the question based on the context.\n\n"
                     f"Context: {context}\n\nQuestion: {question}", 64)

def focus_text(passage, answer=None):
    """The sentence containing the answer plus the next one (cleaner than a raw 100-word window)."""
    sents = re.split(r"(?<=[.!?])\s+", passage)
    if answer:
        for i, s in enumerate(sents):
            if answer in s:
                return " ".join(sents[i:i + 2])
    return passage

def clean_summary(text):
    """Drop repeated sentences; return '' if what is left is too short to be a real summary."""
    seen, out = set(), []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        key = s.lower().strip(" .")
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    result = " ".join(out)
    return result if len(result.split()) >= 5 else ""

def _answer_sentence(passage, answer):
    """Only the sentence that contains the answer, e.g. 'Net income was $30.5 billion, or ...'"""
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", passage):
        if answer and answer in s:
            return s.strip()
    return None

def summarize(text, answer=None, question=None):
    """One clear sentence that states the answer.
    If FLAN-T5's output is vague or contains a number that is not in the filing,
    the filing's own sentence is returned instead."""
    key = _answer_sentence(text, answer)

    # No answer to anchor on (e.g. the executive-summary demo): keep the old behaviour
    if key is None:
        text = " ".join(focus_text(text, answer).split()[:200])
        raw = _generate(f"Summarize the following text in one or two sentences.\n\n{text}", 80,
                        num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.3, early_stopping=True)
        return clean_summary(raw)

    # Ask a specific question about ONE sentence instead of "summarize this"
    if question:
        prompt = (f"Answer the question in one complete sentence using the text.\n\n"
                  f"Question: {question}\nText: {key}\nAnswer:")
    else:
        prompt = (f"Rewrite this sentence as one short, clear sentence that keeps the figure {answer}.\n\n"
                  f"Sentence: {key}")
    raw = clean_summary(_generate(prompt, 60, num_beams=4, no_repeat_ngram_size=3, early_stopping=True))

    # Accept it only if it states the answer's number and invents no new numbers
    answer_nums = numbers_in(answer)
    ok = (raw
          and answer_nums & numbers_in(raw)
          and numbers_in(raw) <= numbers_in(key))
    if ok:
        return raw

    # Fallback: the filing's own sentence (trimmed if very long). Always accurate, never vague
    words = key.split()
    return key if len(words) <= 45 else " ".join(words[:45]) + " …"

# ---------- Evaluation ----------
def judge(answer, gold, context):
    first = first_number(answer)          # 10-Ks list the latest year first
    return {"correct": first in gold,
            "wrong_number": first is not None and first not in gold,
            "no_number": first is None,
            "invented_number": bool(numbers_in(answer) - numbers_in(context))}

def summarize_mode(title, rows):
    n = len(rows)
    lines = [f"\n=== {title} ({n} questions) ==="]
    for model in ("extractive", "generative"):
        c = lambda key: sum(r[model][key] for r in rows)
        lines.append(f"{model.capitalize():11s} correct {c('correct')}/{n} | picked a wrong number {c('wrong_number')} | "
                     f"no number {c('no_number')} | number not in passage {c('invented_number')}")
    return "\n".join(lines)

def evaluate(qa_path="eval/qa_eval.json", k=5):
    items = json.load(open(qa_path, encoding="utf-8"))
    retrievers, pipeline_rows, oracle_rows, hits, missing = {}, [], [], 0, []
    for it in items:
        f, q = it["filing"], it["question"]
        if f not in retrievers:
            retrievers[f] = Retriever(open(f"data/{f}.txt", encoding="utf-8").read())
        r = retrievers[f]
        gold = {g.replace(",", "") for g in it["answers"]}

        # A) End-to-end: retrieve passages, then read them
        hit = any(gold & numbers_in(t["text"]) for t in r.top(q, k))
        hits += hit
        ext = extractive_answer(q, r, k)
        gen = generative_answer(q, ext["context"])
        pipeline_rows.append({"question": q, "gold": it["answers"], "retrieval_hit": hit,
                              "extractive": {"answer": ext["answer"], **judge(ext["answer"], gold, ext["context"])},
                              "generative": {"answer": gen, **judge(gen, gold, ext["context"])}})

        # B) Oracle passage: the passage that contains the gold number, so only the reader is tested
        orc = r.oracle(q, gold)
        if orc is None:
            missing.append(q)
            continue
        e2 = extractive_on_passage(q, orc["text"])
        g2 = generative_answer(q, orc["text"])
        oracle_rows.append({"question": q, "gold": it["answers"],
                            "extractive": {"answer": e2["answer"], **judge(e2["answer"], gold, orc["text"])},
                            "generative": {"answer": g2, **judge(g2, gold, orc["text"])}})

        print(f"\nQ: {q}\n  gold: {it['answers']}   (retrieval found it in top-{k}: {hit})\n"
              f"  [end-to-end] extractive: {ext['answer']!r} | generative: {gen!r}\n"
              f"  [oracle]     extractive: {e2['answer']!r} | generative: {g2!r}")

    n = len(items)
    summary = f"\nRetrieval: right passage in top-{k} for {hits}/{n} questions"
    summary += summarize_mode("END-TO-END (retrieval + reader)", pipeline_rows)
    summary += summarize_mode("ORACLE PASSAGE (reader only)", oracle_rows)
    if missing:
        summary += f"\n\nWARNING: gold number not found anywhere in the filing text for: {missing}"
    print(summary)

    json.dump({"end_to_end": pipeline_rows, "oracle": oracle_rows},
              open("eval/qa_results.json", "w", encoding="utf-8"), indent=2)
    open("eval/qa_summary.txt", "w", encoding="utf-8").write(summary)

    # Executive-summary demo on narrative text (what FLAN-T5 is actually suited for)
    r = next(iter(retrievers.values()))
    passage = r.top("management discussion of risks and outlook", 1)[0]["text"]
    print("\nExecutive-summary demo (FLAN-T5)\nPassage:", passage[:300], "...\nSummary:", summarize(passage))

if __name__ == "__main__":
    evaluate()
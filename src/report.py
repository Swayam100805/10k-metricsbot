import json

items = json.load(open("eval/qa_eval.json", encoding="utf-8"))
kind = {it["question"]: it.get("type", "table") for it in items}
res = json.load(open("eval/qa_results.json", encoding="utf-8"))

for mode, title in (("end_to_end", "END-TO-END"), ("oracle", "ORACLE PASSAGE")):
    for t in ("table", "prose"):
        rows = [r for r in res[mode] if kind.get(r["question"]) == t]
        if not rows:
            continue
        n = len(rows)
        e = sum(r["extractive"]["correct"] for r in rows)
        g = sum(r["generative"]["correct"] for r in rows)
        gi = sum(r["generative"]["invented_number"] for r in rows)
        ei = sum(r["extractive"]["invented_number"] for r in rows)
        print(f"{title:15s} {t:6s} n={n:2d} | extractive {e}/{n} | generative {g}/{n} | "
              f"numbers not in passage: extractive {ei}, generative {gi}")
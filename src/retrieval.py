import re, math
from collections import Counter

STOP = set("""what were was the in of for did does is are a an and to by from at on with as
how much many during year fiscal s apple apples bank america americas company corporation""".split())

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
NUMERIC_Q = re.compile(r"\b(sales|income|revenue|earnings|per share|margin|percentage|percent|ratio|"
                       r"assets|expense|expenses|loss|losses|deposits|billion|million|how much|how many)\b")

def numbers_in(s):
    """Normalised numbers in a string: '416,161' -> '416161', '30.5' -> '30.5'."""
    return {m.rstrip(",").replace(",", "") for m in NUM.findall(s)}

def first_number(s):
    """First number in a string, normalised ('$ 416,161 6 %' -> '416161'); None if there is none."""
    m = NUM.search(s)
    return m.group(0).rstrip(",").replace(",", "") if m else None

def is_numeric_question(q):
    return bool(NUMERIC_Q.search(q.lower()))

def tokenize(s):
    """Keywords plus word pairs, so 'net sales' and 'operating income' match as phrases."""
    words = [w.strip(".,") for w in re.findall(r"[a-z0-9\.,%\$]+", s.lower())]
    uni = [w for w in words if w and w not in STOP]
    bi = [f"{a}_{b}" for a, b in zip(uni, uni[1:])]
    return uni + bi

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}

    def scores(self, query):
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out.append(s)
        return out

class Retriever:
    """Index a filing once, then retrieve passages for many questions.
    Small chunks are used to find the answer; the reader gets the chunk plus a little
    surrounding text (a 'window') so table rows are not cut in half."""
    def __init__(self, text, size=50, stride=25, pad=25):
        self.words = text.split()
        self.size, self.pad = size, pad
        self.starts = list(range(0, max(1, len(self.words) - size + stride), stride))
        self.chunks = [" ".join(self.words[s:s + size]) for s in self.starts]
        self.bm25 = BM25([tokenize(c) for c in self.chunks])

    def window(self, i):
        s = self.starts[i]
        return " ".join(self.words[max(0, s - self.pad): s + self.size + self.pad])

    def _ranked(self, question):
        sc = self.bm25.scores(tokenize(question))
        if is_numeric_question(question):      # figures live in number-heavy passages
            sc = [s * (1 + 0.15 * min(len(NUM.findall(c)), 8)) if len(NUM.findall(c)) >= 3 else s * 0.3
                  for s, c in zip(sc, self.chunks)]
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return order, sc

    def top(self, question, k=5):
        order, sc = self._ranked(question)
        best = sc[order[0]] or 1.0
        return [{"text": self.window(i), "score": sc[i] / best} for i in order[:k]]

    def oracle(self, question, gold_numbers):
        """Best-ranked passage that really contains the gold number (isolates the reader models)."""
        order, _ = self._ranked(question)
        for i in order:
            if gold_numbers & numbers_in(self.window(i)):
                return {"text": self.window(i), "score": 1.0}
        return None
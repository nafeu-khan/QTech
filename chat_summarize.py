
from __future__ import annotations
import argparse
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOP_WORDS = set(stopwords.words('english'))

TOKEN_RE = re.compile(r"[A-Za-z-]+")


def parse_chat(text: str) -> Tuple[List[str], List[str]]:
    user_msgs, ai_msgs = [], []
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("user:"):
            user_msgs.append(line.split(":", 1)[1].strip())
        elif line.lower().startswith("ai:"):
            ai_msgs.append(line.split(":", 1)[1].strip())
    return user_msgs, ai_msgs


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in TOKEN_RE.findall(text) if w.lower() not in STOP_WORDS]
def keyword_stats_simple(user_msgs: List[str], ai_msgs: List[str], k: int = 5) -> List[Tuple[str, int]]:
    all_tokens = []
    for msg in user_msgs + ai_msgs:
        all_tokens.extend(tokenize(msg))
    return Counter(all_tokens).most_common(k)



def keyword_stats_tfidf(user_msgs: List[str], ai_msgs: List[str], k: int = 5) -> List[Tuple[str, float]]:
    # print("user msgs:", user_msgs)
    docs = [" ".join(user_msgs), " ".join(ai_msgs)]  
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(docs)
    # print(vec.get_feature_names_out())
    scores = tfidf.sum(axis=0).A1  
    vocab = vec.get_feature_names_out()
    pairs = list(zip(vocab, scores))
    # print("pairs:", pairs)
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

def summarize(path: Path, use_tfidf: bool = False) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    user_msgs, ai_msgs = parse_chat(text)

    total_msg = len(user_msgs) + len(ai_msgs)
    keywords =keyword_stats_simple(user_msgs, ai_msgs)

    topic = keywords[0][0] if keywords else "general"

    top_keywords = [kw for kw, _ in keywords]
 
    print("Summary:")
    print(f" - The conversation had {total_msg} exchanges.")
    print(f" - The user asked mainly about {topic} and its uses.")
    print(f" - Most common keywords: {', '.join(top_keywords)}.")
    print()


def collect_txt_files(path_str: str) -> List[Path]:
    p = Path(path_str)
    if p.is_file() and p.suffix.lower() == ".txt":
        return [p]
    if p.is_dir():
        return sorted(Path(p).glob("*.txt"))
    raise FileNotFoundError(f"No .txt file or directory found at '{p}'")

def main() -> None:
    ap = argparse.ArgumentParser(description="summarize chat logs")
    ap.add_argument("path", help="path of .txt file or a folder of .txt files if you have multiple files")
    ap.add_argument("--tfidf", action="store_true", help="use TF-IDF library")

    args = ap.parse_args()
    files = collect_txt_files(args.path)
    if not files:
        logger.info("no .txt files found.")
        return
    for f in files:
        print(f"Summarizing {f}...")
        summarize(f, use_tfidf=args.tfidf)


if __name__ == "__main__":
    main()

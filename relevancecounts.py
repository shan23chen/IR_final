"""
Script to get counts of documents with relevance score annotation.
Author: Chester Palen-Michel
"""
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Counter, Dict

from utils import load_clean_wapo_with_embedding


def relevance_counts(path_to_wapo_json_lines: str, topic_number: str):
    counts = defaultdict(int)
    for doc in load_clean_wapo_with_embedding(path_to_wapo_json_lines):
        annotationstr = doc["annotation"]
        if annotationstr.startswith(topic_number):
            fields = annotationstr.split("-")
            topic = fields[0]
            assert topic == topic_number, "topic mismatch"
            counts[fields[1]] += 1
    for rel, count in counts.items():
        print(f"{rel}\t{count}")


def sorting_func(topic: str, counts_by_topic: Dict[str, Counter]) -> float:
    total = 0
    relevant_docs = 0
    for rel, val in counts_by_topic[topic].items():
        total += val
        if rel != "0":
            relevant_docs += val
    return relevant_docs / total if total else 0


def all_relevance_counts(wapo_path: str):
    counts_by_topic = defaultdict(Counter)
    for doc in load_clean_wapo_with_embedding(wapo_path):
        annotationstr = doc["annotation"]
        fields = annotationstr.split("-")
        if len(fields) != 2:
            # No annotation or malformed annotation, skip
            continue
        topic = fields[0]
        counts_by_topic[topic][fields[1]] += 1
    # TODO
    for topic in sorted(
        counts_by_topic, key=lambda x: sorting_func(x, counts_by_topic), reverse=True
    ):
        print(f"Topic: #{topic}")
        for rel, count in counts_by_topic[topic].items():
            print(f"\t{rel}\t{count}")


def dump_text_by_relevance(
    path_to_wapo_json_lines: str, outdir: str, topic_str: str
) -> None:
    texts_by_relevance = defaultdict(list)
    for doc in load_clean_wapo_with_embedding(path_to_wapo_json_lines):
        annotationstr = doc["annotation"]
        fields = annotationstr.split("-")
        if len(fields) != 2:
            # No annotation or malformed annotation, skip
            continue
        topic = fields[0]
        if topic != topic_str:
            continue
        relevance = fields[1]
        texts_by_relevance[relevance].append(doc["content_str"])
    for rel in texts_by_relevance:
        with open(
            os.path.join(outdir, f"rel-doc-text-{rel}.txt"), "w", encoding="utf8"
        ) as outfile:
            outfile.write("\n\n".join(texts_by_relevance[rel]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("wapo_path")
    parser.add_argument("--topic-number")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--dump-docs-dir", help="Directory to dump documents' text by relevance"
    )
    args = parser.parse_args()
    if args.dump_docs_dir:
        if not args.topic_number:
            raise ValueError("Must specify topic number")
        dump_text_by_relevance(args.wapo_path, args.dump_docs_dir, args.topic_number)
    else:
        if args.topic_number and not args.run_all:
            relevance_counts(args.wapo_path, args.topic_number)
        elif args.run_all and not args.topic_number:
            all_relevance_counts(args.wapo_path)

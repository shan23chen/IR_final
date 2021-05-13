"""
Script to summarize the sentences for the wapo corpus.
Authors: Shan Chen (and others??) originally, later heavily edited by Chester Palen-Michel
Chester mostly rewrote to use multiprocessing so could try to run on department machines
over the 50k document subset.
"""
import json
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool, get_context
from typing import Dict, List, Iterable

from transformers import pipeline

from embedding_service.client import EmbeddingClient


def read_docs(wapo_path: str) -> List[Dict]:
    """
    Read docs into memory. For sake of keeping processing simple and since the 50k docs
    aren't that large, we'll just read them in all at once.
    """
    with open(wapo_path, "r", encoding="utf8") as wapo_file:
        return [json.loads(line) for line in wapo_file]


def process_docs(
    docs: Iterable[Dict], default_summarizer, processes: int = 1
) -> List[Dict]:
    process_doc_partial = partial(
        process_doc,
        default_summarizer=default_summarizer,
    )
    new_docs = []
    with get_context("spawn").Pool(processes=processes) as p:
        for i, new_doc in enumerate(
            p.imap_unordered(process_doc_partial, docs, chunksize=15)
        ):
            new_docs.append(new_doc)
            if i % 50 == 0:
                print(i, " docs completed")
    return new_docs


def process_doc(doc: Dict, default_summarizer) -> Dict:
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    default_text, trained_text = sub_summarization(
        doc["content"],
        1024,
        default_summarizer,
    )
    # print("pegasus_summary: ", trained_text)
    doc["summary"] = " ".join(default_text)
    doc["summary_vector"] = encoder.encode(default_text).tolist()[0]
    # doc['pegasus_summary'] = trained_text
    return doc


def sub_summarization(para, k, default_sum):
    """
    It's unclear to me why they decided to split on 1024 characters.
    Ideally this should preserve token boundaries and sentence
    boundaries, but it doesn't, and the summaries aren't that bad in spite
    of how things are being divided up.
    """
    default_text = []
    trained_text = []
    for i in range(len(para) // k + 1):
        if i < len(para) // k:
            batch = para[i * 1024 : (i + 1) * 1024]
        else:
            batch = para[i * 1024 : len(para)]
        default_text.append(
            default_sum(batch, max_length=150, min_length=2)[0]["summary_text"]
        )
        # trained_text.append(pre_trained_sum(batch, max_length=150, min_length=2)[0]['summary_text'])
    return default_text, trained_text


def dump_corpus(docs: Iterable[Dict], outpath: str) -> None:
    with open(outpath, "w", encoding="utf8") as outfile:
        for doc in docs:
            print(json.dumps(doc), file=outfile)


def summarize():
    parser = ArgumentParser()
    parser.add_argument("wapo_path", help="path to wapo json lines")
    parser.add_argument("outpath", help="path to dump the new wapo json lines")
    parser.add_argument(
        "--num-processes", default=1, type=int, help="Number of processes to use"
    )
    args = parser.parse_args()

    default_summarizer = pipeline("summarization")
    # # Original plan was to do two different summary models, but pegasus was too slow.
    # pegasus_summarizer = pipeline(
    #     'summarization', model="google/pegasus-xsum", tokenizer="google/pegasus-xsum"
    # )

    docs = read_docs(args.wapo_path)
    processed_docs = process_docs(
        docs, default_summarizer, processes=args.num_processes
    )
    dump_corpus(processed_docs, args.outpath)


if __name__ == "__main__":
    summarize()

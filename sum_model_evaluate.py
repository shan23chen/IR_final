"""
2021-05
@Xiangyu Li
@Shan Chen
We use the default BM25 method of elastic search to find related documents,
and sort by the sum of the cosine similarity between each subsection of the document and the query
"""

import argparse

import numpy as np
from elasticsearch import Elasticsearch
from embedding_service.client import EmbeddingClient
from metrics import Score
from utils import parse_wapo_topics

es = Elasticsearch()

def search(topic_id, index, k, q):
    result_annotations = []
    # use bert to encode
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    query_vector = encoder.encode([q], pooling="mean").tolist()[0]
    # get result by searching content and title
    content_result = es.search(index=index, size=k, body={"query": {"match": {"content": q}}})
    title_result = es.search(index=index, size=k, body={"query": {"match": {"title": q}}})  # todo
    # calculate cosine similarity
    doc_list = {}
    for doc in content_result['hits']['hits']+title_result['hits']['hits']:
        embed_vec_list = np.array(doc['_source']['sbert_vector'])
        doc_list[doc['_id']] = np.max(np.dot(embed_vec_list, np.array(query_vector)))
    ordered_doc = sorted(doc_list.items(), key=lambda kv: (kv[1], kv[0]))
    ordered_doc.reverse()
    result_list = [i[0] for i in ordered_doc]
    print(result_list)
    # find the result's annotation
    for i in ordered_doc:
        if es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[0] == topic_id:
            result_annotations.append(int(es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[1]))
        else:
            result_annotations.append(0)
    print(result_annotations)
    return result_annotations


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", required=True, type=str, help="name of the ES index")
    parser.add_argument("--topic_id", required=True, type=int, help="topic id")
    parser.add_argument("--query_type", required=True, type=str, choices=["title", "narration", "description"],
                        help="title, narration, description")
    parser.add_argument("--top_k", required=True, type=int, help="top k")
    return parser.parse_args()


def dot(l1, l2):
    return sum(a*b for a, b in zip(l1, l2))


def cosine_similarity(a, b):
    return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))


if __name__ == "__main__":
    query_type_index = {"title": 0, "description": 1, "narration": 2}
    args = build_args()
    query = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][query_type_index[args.query_type]]
    searched_result = search(str(args.topic_id), args.index_name, args.top_k, query)
    score = Score
    print(score.eval(searched_result, args.top_k))

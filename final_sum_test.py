from elasticsearch import Elasticsearch
from embedding_service.client import EmbeddingClient
import numpy as np
from metrics import Score
from scipy import spatial
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
import argparse
from utils import parse_wapo_topics


es = Elasticsearch()

def cosim(result, type, query, content):
    cos_score = {}
    if type == "sbert":
        fieldName = "sbert_vector"
        encoder = EmbeddingClient(
            host="localhost", embedding_type="sbert"
        )  # connect to the sbert embedding server
    else:
        fieldName = "ft_vector"
        encoder = EmbeddingClient(
            host="localhost", embedding_type="fasttext"
        )  # connect to the fasttext embedding server
    embedding = encoder.encode(query)

    print(embedding.shape)
    for i in range(len(result)):
        doc_vec = np.array(content[result[i]])
        current_calc = spatial.distance.cosine(embedding, doc_vec)
        cos_score[result[i]] = current_calc
        # print(current_calc, np.array(content['_source'][fieldName]).shape)

    print(cos_score)
    sorted_string = sorted(cos_score.items(), key=lambda item: item[1])
    final = [id[0] for id in sorted_string]

    return final

# # use title from topic 321 as the query; search over the custom_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
# python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type title -u --top_k 20
#
# # use narration from topic 321 as the query; search over the content field from index "wapo_docs_50k" based on sentence BERT embedding and compute NDCG@20
# python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type narration --vector_name sbert_vector --top_k 20

# get query
def get_query_from_TREC(id, query_type):
    if query_type == "title":
        return(parse_wapo_topics("pa5_data/topics2018.xml")[str(id)][0])
    elif query_type == "narration":
        print("n1")
        return(parse_wapo_topics("pa5_data/topics2018.xml")[str(id)][2])
    else:
        return(parse_wapo_topics("pa5_data/topics2018.xml")[str(id)][1])


# get result func

def results(topic_id, top_k, query_type, vector_name, index_name, custom):

    query_text = get_query_from_TREC(topic_id, query_type)

    result_list = []
    result_list.clear()

    id2vec_ft = {}
    id2vec_sbert = {}

    q_match_all = MatchAll()

    query_results = es.search(index=index_name, size=top_k, body={"query": {"match": {"content": query_text}}})
    print(query_results)


    for doc in query_results["hits"]["hits"]:
        result_list.append(doc["_id"])
        id2vec_ft[doc["_id"]] = doc['_source']['ft_vector']
        id2vec_sbert[doc["_id"]] = doc['_source']['sbert_vector']
    result_list = result_list[:top_k]

    if vector_name == "ft_vector" or vector_name == "sbert_vector":

        if vector_name == "ft_vector":
            print("3", vector_name)

            result_list = cosim(result_list, "fastText", [query_text], id2vec_ft)

        else:
            print("4", vector_name)
            # content = es.get(index = "wapo_docs_50k", id = str(result_list[0]),doc_type="_all") #dict type
            match = []
            result_list = cosim(result_list, "sbert", [query_text], id2vec_sbert)

    else:
        print(repr(custom))
        if custom == "True":
            print("1")
            result_list = []
            query_results2 = es.search(index=index_name, size=top_k, body={"query": {"match": {"custom_content": query_text}}})
            for doc in query_results2["hits"]["hits"]:
                result_list.append(doc["_id"])
            result_list = result_list[:top_k]

        else:
            print("2")

    result = []
    for info in result_list:
        content = es.get(index=index_name, id=str(info), doc_type="_all")

        if (content['_source']['annotation']).rsplit("-")[0] == topic_id:
            result.append(int((content['_source']['annotation']).rsplit("-")[1]))
        else:
            result.append(0)
    print(result_list)
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name",
        required=True,
        type=str,
        help="name of the ES index",
    )
    parser.add_argument(
        "--topic_id",
        required=False,
        type=str,
        help="topic_id",
    )
    parser.add_argument(
        "--query_type",
        required=False,
        type=str,
        choices=["narration", "title", "description"],
        help="choose search fields",
    )
    parser.add_argument(
        "--top_k",
        required=False,
        type=int,
        help="choose how many documents",
    )
    parser.add_argument(
        "--vector_name",
        required=False,
        type=str,
        choices=["sbert_vector", "ft_vector"],
        help="choose which embedding you are using",
    )
    parser.add_argument(
        "--custom",
        '-u',
        required=False,
        action='store_true',
        help="custom or not",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = main()
    # xxx = []
    x = ["title","description","narration"]
    # for id in x:

    # result = results(str(args.topic_id), int(args.top_k), str(args.query_type), str(args.vector_name), str(args.index_name), str(args.custom))
    # s = Score
    # print(s.eval(result, args.top_k))
    # print(str(args.query_type))
    # for id, result in zip(x,xxx):
    #     s = Score
    #     print(id,": ", s.eval(result, args.top_k))

    for i in x:
        print(i)
        result = results(str(args.topic_id), int(args.top_k), str('invasive'), str(args.vector_name), str(args.index_name), str(args.custom))
        s = Score
        print(s.eval(result, args.top_k))




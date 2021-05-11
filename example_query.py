from typing import List

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Match, MatchAll, ScriptScore, Ids, Query
from elasticsearch_dsl.connections import connections
from embedding_service.client import EmbeddingClient

encoder = EmbeddingClient(host="localhost", embedding_type="sbert")


def generate_script_score_query(query_vector: List[float], vector_name: str) -> Query:
    """
    generate an ES query that match all documents based on the cosine similarity
    :param query_vector: query embedding from the encoder
    :param vector_name: embedding type, should match the field name defined in BaseDoc ("ft_vector" or "sbert_vector")
    :return: an query object
    """
    q_script = ScriptScore(
        query={"match_all": {}},  # use a match-all query
        script={  # script your scoring function
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            # add 1.0 to avoid negative score
            "params": {"query_vector": query_vector},
        },
    )
    return q_script


def search(index: str, query: Query) -> None:
    s = Search(using="default", index=index).query(query)[
        :5
    ]  # initialize a query and return top five results
    response = s.execute()
    for hit in response:
        print(
            hit.meta.id, hit.meta.score, hit.title, sep="\t"
        )  # print the document id that is assigned by ES index, score and title


if __name__ == "__main__":
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

    q_match_all = MatchAll()  # a query that matches all documents
    q_basic = Match(
        title={"query": "D.C"}
    )  # a query that matches "D.C" in the title field of the index, using BM25 as default
    q_match_ids = Ids(values=[1, 3, 2])  # a query that matches ids

    query_text = ["students pursue college education"]
    query_vector = encoder.encode(query_text, pooling="mean").tolist()[
        0
    ]  # get the query embedding and convert it to a list
    q_vector = generate_script_score_query(
        query_vector, "sbert_vector"
    )  # custom query that scores documents based on cosine similarity

    q_c = (
        q_match_ids & q_basic
    )  # you can also have a compound query by using logic operators on multiple queries

    search(
        "wapo_docs_50k", q_match_all
    )  # search, change the query object to see different results

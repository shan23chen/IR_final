#! /usr/bin/env python
"""
Script to evaluate queries for IR project.
This version built off ta solution and supports running all queries and outputting scores
to tsv
Author: Chester Palen-Michel
"""
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict, Union

from attr import attrs
from elasticsearch_dsl import Search
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl.query import Match, ScriptScore

from embedding_service.client import EmbeddingClient
from metrics import Score
from wikidata_expander import WikidataExpander
from utils import parse_wapo_topics
from wordnetexpander import WordnetExpander, WSD

QUERY_DATA_PATH = "pa5_data/topics2018.xml"
VECTOR_CHOICES = [
    "ft_vector",
    "sbert_vector",
]
WORDNET_EXPANSION_CHOICES = frozenset({"FIRST", "LESK", "BERT"})
INDEX = "wapo_docs_50k"
BERT_ENCODER = EmbeddingClient(host="localhost", embedding_type="sbert")
FT_ENCODER = EmbeddingClient(host="localhost", embedding_type="fasttext")


@attrs(frozen=True, auto_attribs=True)
class Queries:
    title: str
    narrative: str
    description: str
    topic_id: str

    def __iter__(self):
        return iter(
            [
                (self.title, "title"),
                (self.narrative, "narrative"),
                (self.description, "description"),
            ]
        )


@attrs(frozen=True, auto_attribs=True)
class Document:
    id: str
    content: str
    topic: str
    relevance: str


@attrs(frozen=True, auto_attribs=True)
class Result:
    """Class to hold results mapping of {approach: score}"""

    approach: str
    query: str
    scores: Score


class ReturnedDocs:
    docs: List[Document]
    query: str


def prepare_topics(wapo_topic_dict: Dict[str, List[str]], topics: List[str]) -> Dict[str, Queries]:
    """
    Reworking the parsed topic dict, because accessing by ordered list is annoying.
    """
    return {
        topic: Queries(
            wapo_topic_dict[topic][0],
            wapo_topic_dict[topic][1],
            wapo_topic_dict[topic][2],
            topic
        )
        for topic in topics
    }


def expand_queries(
    expander: Union[WordnetExpander, WikidataExpander], wapo_topics: Dict[str, Queries]
) -> Dict[str, Queries]:
    """
    Expand each type of query using the specified expander and return in same
    topic-queries mapping.
    """
    ret = {}
    for topic in wapo_topics:
        print(topic)
        ret[topic] = Queries(
            expander.expand(wapo_topics[topic].title),
            expander.expand(wapo_topics[topic].narrative),
            expander.expand(wapo_topics[topic].description),
            topic,
        )
    return ret


def dump_queries(queries_by_topic: Dict[str, Queries], outpath: str):
    with open(outpath, "w", encoding="utf8") as outfile:
        print("\t".join(["topic", "title", "narrative", "description"]), file=outfile)
        for topic in queries_by_topic:
            queries = queries_by_topic[topic]
            print(
                "\t".join(
                    [topic, queries.title, queries.narrative, queries.description]
                ),
                file=outfile,
            )


def process_annotation(annotation: str, topicid: str):
    if annotation:
        fields = annotation.split("-")
        if len(fields) != 2:
            return 0
        else:
            if fields[0] == topicid:
                return int(fields[-1])
            else:
                return 0
    else:
        # In the case that the document is unlabeled, which seems to be the case in some examples
        return 0


def responses_to_relevance(response, topic_id):
    return [process_annotation(hit.annotation, topic_id) for hit in response]


def run_bm25(query: str, topic_id: str, custom=False, top_k: int = 20) -> Score:
    if custom:
        q = Match(custom_content={"query": query})
    else:
        q = Match(content={"query": query})
    s = Search(using="default", index=INDEX).query(q)[:top_k]
    responses = s.execute()
    relevance = responses_to_relevance(responses, topic_id)
    return Score.eval(relevance, top_k)


def run_vector_query(
    query: str, vec_name: str, encoder, topic_id: str, top_k: int = 20
) -> Score:
    q_bm25 = Match(content={"query": query})
    s = Search(using="default", index=INDEX).query(q_bm25)[:top_k]

    query_vector = encoder.encode([query], pooling="mean").tolist()[0]
    q_vector = generate_script_score_query(query_vector, vector_name=vec_name)
    s = s.extra(
        rescore={
            "window_size": top_k,
            "query": {
                "rescore_query": q_vector,
                "query_weight": 0,
                "rescore_query_weight": 1,
            },
        }
    )
    responses = s.execute()
    relevance = responses_to_relevance(responses, topic_id)
    return Score.eval(relevance, top_k)

def generate_script_score_query(query_vector, vector_name: str):
    q_script = ScriptScore(
        query={"match_all": {}},
        script={
            "source": f"cosineSimilarity(params.query_vector, '{vector_name}') + 1.0",
            "params": {"query_vector": query_vector},
        },
    )
    return q_script

def execute_query(queries: Queries, top_k: int) -> List[Result]:
    # TODO: run each type of query
    ret = []
    for query, query_name in queries:
        ret.append(
            Result("bm_25", query_name, run_bm25(query, queries.topic_id, top_k=top_k))
        )
        ret.append(
            Result(
                "custom",
                query_name,
                run_bm25(query, queries.topic_id, custom=True, top_k=top_k),
            )
        )
        ret.append(
            Result(
                "bert",
                query_name,
                run_vector_query(
                    query, "sbert_vector", BERT_ENCODER, queries.topic_id, top_k=top_k
                ),
            )
        )
        ret.append(
            Result(
                "fast",
                query_name,
                run_vector_query(
                    query, "ft_vector", FT_ENCODER, queries.topic_id, top_k=top_k
                ),
            )
        )
    return ret


def run_queries(
    queries_by_topic: Dict[str, Queries], topics: List[str], top_k: 20
) -> Dict[str, List[Result]]:
    return {topic: execute_query(queries_by_topic[topic], top_k) for topic in topics}


def dump_results(results_by_topic: Dict[str, List[Result]], outpath: str):
    with open(outpath, 'w', encoding='utf8') as outfile:
        # HEADER
        print("\t".join(["Topic", "Query.type", "Approach", "precision",
                         "Avg.Precision", "NDCG"]), file=outfile)
        for topic in results_by_topic:
            for result in results_by_topic[topic]:
                print("\t".join([topic, result.query, result.approach, str(result.scores.prec),
                                 str(result.scores.ap), str(result.scores.ndcg)]), file=outfile)


def average_across_queries(results_by_topic: Dict[str, List[Result]]):
    print("Avg scores across queries")
    strat_scores = defaultdict(list)
    for topic in results_by_topic:
        for r in results_by_topic[topic]:
            strat_scores[r.approach + "-" + r.query].append(r.scores.ndcg)
    for strat in strat_scores:
        print(strat, sum(strat_scores[strat]) / len(results_by_topic))


def run_evaluate():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--topics", nargs="+")
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument(
        "--expansion",
        choices=["DEFAULT", "FIRST", "LESK", "BERT", "WIKIDATA"],
        help="Strategy for expansion",
    )
    parser.add_argument("--dump-queries", help="path to dump expanded queries to")
    parser.add_argument(
        "--dump-scores", help="Run scores and dump results to this path"
    )
    args = parser.parse_args()

    # Things to init
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    wiki_expander = WikidataExpander()
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

    # topicid : [title, narrative, description]
    wapo_topics = prepare_topics(parse_wapo_topics(QUERY_DATA_PATH), args.topics)

    # Expand query topics
    if args.expansion == "DEFAULT":
        print("Skipping expansion")
        queries_by_topic = wapo_topics
    elif args.expansion in WORDNET_EXPANSION_CHOICES:
        wn_expander = WordnetExpander(
            wsd_strategy=WSD.string_to_enum(args.expansion),
            encoder=encoder,
        )
        print(f"Expanding with wordnet using {args.expansion}")
        queries_by_topic = expand_queries(wn_expander, wapo_topics)
    elif args.expansion == "WIKIDATA":
        print("Expanding using wikidata")
        queries_by_topic = expand_queries(wiki_expander, wapo_topics)
    else:
        raise ValueError("Unrecognized expansion choice")

    # Dump expanded queries for review if desired
    if args.dump_queries:
        dump_queries(queries_by_topic, args.dump_queries)

    # Run each query approach on each topic
    if args.dump_scores:
        results_by_topic = run_queries(queries_by_topic, args.topics, args.topk)
        dump_results(results_by_topic, args.dump_scores)
        average_across_queries(results_by_topic)


if __name__ == "__main__":
    run_evaluate()

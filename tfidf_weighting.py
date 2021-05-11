# 方法1
import heapq
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from elasticsearch_dsl import analyzer, connections
from scipy.special import softmax
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import argparse

from elasticsearch import Elasticsearch, helpers

from embedding_service.client import EmbeddingClient
from metrics import Score
from utils import parse_wapo_topics
es = Elasticsearch()


def search(topic_id, index, k, model, vector, q, analyzer):
    doc_ids = []
    encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")

    # # transfer query to custom style
    # q_response = analyzer.simulate(q)
    # custom_q = " ".join([t.token for t in q_response.tokens])

    # build query vector
    query_vector = 5 * np.array(doc_embedding(10, model, q)) + encoder.encode([q], pooling="mean").tolist()[0]

    # query_vector = doc_embedding(40, model, custom_q)
    # get k results by using customized query
    # result = es.search(index=index, size=k, body={"query": {"match": {"custom_content": custom_q}}})

    result = es.search(index=index, size=150, body={"query": {"match": {"content": q}}})

    # ranking by embedding
    doc_list = {}
    for doc in result['hits']['hits']:
        # print(doc['_source']['annotation'])
        # transfer content to customized one
        # response = analyzer.simulate(doc['_source']['content'])
        # custom_content = " ".join([t.token for t in response.tokens])
        # using top 20 largest tfidf term for doc

        # if doc['_source']['title']:
        #     embed_vec = 5 * np.array(doc_embedding(10, model, doc['_source']['content'])) + np.array(doc['_source']['ft_vector']) + 2 * np.array(doc_embedding(10, model, doc['_source']['title']))
        # else:

        embed_vec = 5 * np.array(doc_embedding(10, model, doc['_source']['content'])) + np.array(doc['_source']['ft_vector'])
        cs = cosine_similarity(query_vector, embed_vec)
        doc_list[doc['_id']] = cs
    ordered_doc = sorted(doc_list.items(), key=lambda kv: (kv[1], kv[0]))
    ordered_doc.reverse()

    # save ranked doc ids
    for i in ordered_doc:
        if es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[0] == topic_id:
            # print(es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-'))
            doc_ids.append(int(es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[1]))
        else:
            doc_ids.append(0)
    print(doc_ids[:20])
    print(doc_ids.count(2))
    print(doc_ids.count(1))
    return doc_ids[:20]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", required=True, type=str, help="name of the ES index")
    parser.add_argument("--topic_id", required=True, type=int, help="topic id")
    parser.add_argument("--query_type", required=True, type=str, choices=["title", "narration", "description"],
                        help="title, narration, description")
    parser.add_argument('--usermode', '-u', required=False, action='store_true', help='custom content')
    parser.add_argument("--vector_name", required=False, type=str, choices=["sbert_vector", "ft_vector"],
                        help="sbert_vector or ft_vector")
    parser.add_argument("--top_k", required=True, type=int, help="top k")
    return parser.parse_args()


def dot(l1, l2):
    return sum(a*b for a, b in zip(l1, l2))


def cosine_similarity(a, b):
    return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))


# keywords embedding
def tokenize(text):
    return [PorterStemmer().stem(item) for item in nltk.word_tokenize(text)]


def build_corpus_model(index, analyzer, es_size):
    # given index_path and topic_id
    # return tf-idf model
    # with open(wapo_jl_path, "r+", encoding="utf8") as f:
    #     contents = []
    #     ids = []
    #     for item in f:
    #         obj = json.loads(item)
    #         contents.append(obj["content_str"])
    #         ids.append(obj["doc_id"])
    # contents = [" ".join([re.sub('[^a-z]', '', word.lower()) for word in file.split() if word not in stopwords.words("english")]) for file in contents]

    # searched = es.search(index=index, size=es_size, body={"query": {"match_all": {}}}, scroll="5m")
    # result = searched['hits']['hits']
    # total = int(searched['hits']['total']['value'])
    # scroll_id = searched['_scroll_id']
    # for i in range(0, int(total / 100) + 1):
    #     # scroll参数必须指定否则会报错
    #     query_scroll = es.scroll(scroll_id=scroll_id, scroll='5m')['hits']['hits']
    #     result += query_scroll

    result = es.search(index=index, size=33, body={
                  "query": {
                    "bool": {
                      "must": [
                          {"match": {"annotation": "690-2"}},
                          {"match": {"annotation": "690-1"}}
                      ],
                    }
                  }
                })
    custom_contents = []
    for doc in result['hits']['hits']:
        # response = analyzer.simulate(doc['_source']['content'])

        content = doc['_source']['content']
        blob = TextBlob(content).noun_phrases

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(" ".join(blob))
        filtered_sentence = [w for w in word_tokens if not w in stop_words and w.isalpha()]
        custom_contents.append(" ".join(filtered_sentence))

        # custom_contents.append(" ".join([t.token for t in response.tokens]))
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfidf.fit_transform(custom_contents)
    return tfidf


def doc_embedding(k, tfidf, doc):
    # given doc_content, tfidf model, and k.
    # return doc's embedding vector
    encoder = EmbeddingClient(host="localhost", embedding_type="fasttext")
    response = tfidf.transform([doc])
    feature_names = tfidf.get_feature_names()
    tfidf_dict = {}
    for col in response.nonzero()[1]:
        tfidf_dict[feature_names[col]] = response[0, col]
    h = []
    for value in tfidf_dict:
        heapq.heappush(h, (tfidf_dict[value], value))
    kw = np.array([i[1] for i in heapq.nlargest(k, h)])

    # # use softmax to weight
    # kw_tfidf = np.array([tfidf_dict[word] for word in kw])
    # softmaxed = softmax(kw_tfidf)
    #
    # doc_vector = np.array(np.zeros(300,))
    # for term, weight in zip(kw, softmaxed):
    #     word_vector = np.array(encoder.encode([term], pooling="mean").tolist()[0])
    #     doc_vector += word_vector*weight
    # return doc_vector
    # sum 20 keywords embedding
    # print(kw)
    kwords = " ".join(kw)
    return encoder.encode([kwords], pooling="mean").tolist()[0]


if __name__ == "__main__":
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")
    custom_analyzer = analyzer(
        "custom_analyzer",
        tokenizer="standard",
        filter=["lowercase", "asciifolding", "snowball", "stop"],

    )

    print("building tfidf model")
    tfidf_model = build_corpus_model("wapo_docs_50k", custom_analyzer, 100)
    print("built tfidf model")

    query_type_index = {"title": 0, "description": 1, "narration": 2}
    args = build_args()
    # query = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][query_type_index[args.query_type]]
    query0 = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][0]
    query1 = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][1]
    query2 = parse_wapo_topics("pa5_data/topics2018.xml")[str(args.topic_id)][2]
    searched_result0 = search(str(args.topic_id), args.index_name, args.top_k, tfidf_model, args.vector_name, query0, custom_analyzer)
    searched_result1 = search(str(args.topic_id), args.index_name, args.top_k, tfidf_model, args.vector_name, query1, custom_analyzer)
    searched_result2 = search(str(args.topic_id), args.index_name, args.top_k, tfidf_model, args.vector_name, query2, custom_analyzer)
    score = Score
    print(score.eval(searched_result0, args.top_k))
    print(score.eval(searched_result1, args.top_k))
    print(score.eval(searched_result2, args.top_k))


# BERT+default analyzer	0.5	0.3869	0.3333

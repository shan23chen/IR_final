"""
2021-05
@Xiangyu Li
@Shan Chen
The final present page
"""

from flask import *
from elasticsearch import Elasticsearch
from embedding_service.client import EmbeddingClient
import numpy as np
from scipy import spatial
from metrics import Score
from utils import parse_wapo_topics

app = Flask(__name__)

es = Elasticsearch()

result_list = []

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

    #get id list:
    result_list = [i[0] for i in ordered_doc]

    # find the result's annotation
    for i in ordered_doc:
        if es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[0] == topic_id:
            result_annotations.append(int(es.get(index=index, id=i[0], doc_type="_all")['_source']['annotation'].split('-')[1]))
        else:
            result_annotations.append(0)
    print(result_annotations)
    return result_annotations, result_list

# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    query_text = request.form["query"]  # Get the raw user query from home page
    topic_id = request.form["method"]
    global result_list
    result_list.clear()
    match = []
    id2vec_ft = {}
    id2vec_sbert = {}

    result_annotations, result_list = search(topic_id, "test2", 20, query_text)

    for info, score in zip(result_list, result_annotations):
        content = es.get(index='test2', id=str(info), doc_type="_all")
        wapo = content['_source']

        title = wapo["title"]
        short_snippet = str(wapo["content"])[:150]

        match.append((info, title, short_snippet, score))

    global length
    length = len(result_list)
    matches = match[:8]
    query = query_text
    max_pages = (len(match) // 8)

    result_list = match
    return render_template('results.html', page=1, matches=matches, query=query_text, max_pages=max_pages, length=length, result_annotations = result_annotations)


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    # TODO:
    #print(page_id,"iii")
    query_text = request.form["query"]
    global length
    match = result_list
    matches = match[page_id * 8:(page_id + 1) * 8]  # show next page result in matching query
    query = query_text
    max_pages = (len(match) // 8)  # tracker to see whether next page button is useful
    match = []  # clean match for reuse to avoid assigned reference error

    #print(max_pages, "ggg")
    return render_template('results.html', page=page_id+1, matches=matches, query=query_text, max_pages=max_pages, length=length)


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    # TODO:
    article = es.get(index='test2', id=str(str(doc_id)), doc_type="_all")['_source'] # just for less typing
    context = str(article["content"])
    return render_template("doc.html", article=article, context=Markup(context), pd=article["date"],
                           author=article["author"], title=article["title"])
    # adding markup to make the context pretty


if __name__ == "__main__":
    app.run(debug=True, port=5001)

from flask import *
from elasticsearch import Elasticsearch
from embedding_service.client import EmbeddingClient
import numpy as np
from scipy import spatial

app = Flask(__name__)

es = Elasticsearch()

result_list = []

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

# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    query_text = request.form["query"]  # Get the raw user query from home page
    method = request.form["method"]
    global result_list
    result_list.clear()
    match = []
    id2vec_ft = {}
    id2vec_sbert = {}

    query_results = es.search(index='wapo_docs_50k', size=20, body={"query": {"match": {"content": query_text}}})

    for doc in query_results["hits"]["hits"]:
        result_list.append(doc["_id"])
        id2vec_ft[doc["_id"]] = doc['_source']['ft_vector']
        id2vec_sbert[doc["_id"]] = doc['_source']['sbert_vector']
    result_list = result_list[:20]

    if method == "BM25d":
        print("1")
        match = []
        print(result_list)

    elif method == "BM25c":
        print("2")
        match = []
        result_list = []
        query_results2 = es.search(index='wapo_docs_50k', size=20, body={"query": {"match": {"custom_content": query_text}}})
        for doc in query_results2["hits"]["hits"]:
            result_list.append(doc["_id"])
        result_list = result_list[:20]
        print(result_list)

    elif method == "fastText":
        print("3")
        match = []
        result_list = cosim(result_list, "fastText", [query_text], id2vec_ft)

    else:
        print("4")
        # content = es.get(index = "wapo_docs_50k", id = str(result_list[0]),doc_type="_all") #dict type
        match = []
        result_list = cosim(result_list, "sbert", [query_text], id2vec_sbert)


    for info in result_list:
        content = es.get(index="wapo_docs_50k", id=str(info), doc_type="_all")
        wapo = content['_source']

        title = wapo["title"]
        short_snippet = str(wapo["content"])[:150]
        match.append((info, title, short_snippet, 0))

    global length
    length = len(result_list)
    matches = match[:8]
    query = query_text
    max_pages = (len(match) // 8)

    result_list = match
    return render_template('results.html', page=1, matches=matches, query=query_text, max_pages=max_pages, length=length)


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
    article = es.get(index="wapo_docs_50k", id=str(str(doc_id)), doc_type="_all")['_source'] # just for less typing
    context = str(article["content"])
    return render_template("doc.html", article=article, context=Markup(context), pd=article["date"],
                           author=article["author"], title=article["title"])
    # adding markup to make the context pretty


if __name__ == "__main__":
    app.run(debug=True, port=5001)

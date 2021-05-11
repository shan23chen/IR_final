"""
2021-05
@Xiangyu Li
@Shan Chen
# get embeddings after Chester get the full dataset with summary
"""

import json
from embedding_service.client import EmbeddingClient

#get embeddings after Chester get full documents of summerization
def sa(input_file, new_jsonfile):
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    allfile = open("all-690-805.jl", "r", encoding="utf-8").readlines()
    with open(input_file, "r", encoding="utf-8") as filted:
        with open(new_jsonfile, "w") as new_json:
            for i, line in enumerate(filted):
                new_doc = {}
                doc = json.loads(line)
                new_doc['title'] = doc['title']
                new_doc['doc_id'] = doc['doc_id']
                new_doc['author'] = doc['author']
                new_doc['annotation'] = doc['annotation']
                new_doc['published_date'] = doc['published_date']
                for dc in allfile:
                    q = json.loads(dc)
                    if q['doc_id'] == new_doc['doc_id']:
                        print("Q")
                        new_doc['content_str'] = q['content_str']
                new_doc['default_text'] = doc['default_text']
                default_text = [str(item[0]) for item in new_doc['default_text']]
                new_doc['default_vector'] = encoder.encode(default_text).tolist()
                json.dump(new_doc, new_json)
                new_json.write('\n')
                print(i, "done")


sa("filted0.json", "test.jl")


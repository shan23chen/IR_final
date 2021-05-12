"""
2021-05
@Xiangyu Li
@Shan Chen
# This is a helper script.
# get embeddings and add content after Chester get the full dataset with summary
# Note:: This script will create a new jl file or not add data to existed one.!!
"""
import json
from embedding_service.client import EmbeddingClient


def main(input_file, output_file):
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    allfiles = open("all-690-805.jl", "r", encoding="utf-8").readlines()

    with open(output_file, "w") as new_json:
        with open(input_file, "r", encoding="utf-8") as filted2:
            for i, line in enumerate(filted2):
                new_doc = {}
                doc = json.loads(line)
                new_doc['title'] = doc['title']
                new_doc['doc_id'] = doc['doc_id']
                for d in allfiles:
                    if json.loads(d)['doc_id'] == new_doc['doc_id']:
                        new_doc['content'] = json.loads(d)['content_str']
                new_doc['author'] = doc['author']
                new_doc['annotation'] = doc['annotation']
                new_doc['published_date'] = doc['published_date']
                new_doc['summary'] = doc['default_text']
                default_text = [str(item[0]) for item in new_doc['summary']]
                new_doc['summary_vector'] = encoder.encode(default_text).tolist()
                json.dump(new_doc, new_json)
                new_json.write('\n')
                print(i, "done")


if __name__ == "__main__":
    main("filted0.json", "same_structure.jl")

"""
2021-05
@Xiangyu Li
@Shan Chen
#get summarization from default and pretrained BERT with rewriting a new json file
Tried using muti thread to improve the speed, it did work but slow, running this script on colab:
"""
import threading
import json
import os
import time

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from embedding_service.client import EmbeddingClient

"""
Grab all documents that are relevant with 690 and 805
"""
def taker(input_file, output_file, index_num):
    with open(input_file, "r", encoding="utf-8") as old_json:
        with open(output_file, "w") as new_json:
            for i, line in enumerate(old_json):
                doc = json.loads(line)
                if doc['annotation'][:3] == str(index_num) or doc['annotation'][:3] == '805':   # index id / hard code
                    json.dump(doc, new_json)
                    new_json.write('\n')


class DataTransfer:
    def __init__(self):
        print("initializing summarizer...")
        self.summarizer = pipeline('summarization')
        print("initializing tokenizer and model...")
        # self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
        print("initialized")
        self.encoder = EmbeddingClient(host="localhost", embedding_type="sbert")

    # default summarizer
    def default_sum(self, min_length, max_length, text):
        return self.summarizer(text[:1024], max_length=max_length, min_length=min_length)[0]['summary_text']

    """
    Too slow to run pretrained on our own machine or colab
    """
    # # pre-trained summarizer
    # def pre_trained_sum(self, text):
    #     inputs = self.tokenizer([text], max_length=1024, return_tensors='pt')
    #     summary_ids = self.model.generate(inputs['input_ids'])
    #     return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    """
        Rewrite Json file
        Tried using muti thread to improve the speed but it did not work:
    """
    def transform_json(self, old_jsonfile, new_jsonfile):
        # delete file if exist
        if os.path.isfile(new_jsonfile):
            os.remove(new_jsonfile)
        with open(old_jsonfile, "r", encoding="utf-8") as old_json:
            with open(new_jsonfile, "w") as new_json:
                for i, line in enumerate(old_json):
                    new_doc = {}
                    doc = json.loads(line)
                    new_doc['title'] = doc['title']
                    print("   subsection...")
                    new_doc['default_text'], new_doc['trained_text'] = self.sub_summarization(doc['content'], 1024)
                    print("   done")
                    new_doc['doc_id'] = doc['doc_id']
                    new_doc['author'] = doc['author']
                    new_doc['annotation'] = doc['annotation']
                    new_doc['published_date'] = doc['published_date']
                    default_text = [str(item[0]) for item in new_doc['default_text']]
                    new_doc['default_vector'] = self.encoder.encode(default_text).tolist()
                    trained_vector = [str(item[0]) for item in new_doc['trained_text']]
                    new_doc['trained_vector'] = self.encoder.encode(trained_vector).tolist()
                    json.dump(new_doc, new_json)
                    new_json.write('\n')
                    print(i, "done")

    # helper function splits text and get summarization
    def sub_summarization(self, para, k):
        default_text = []
        trained_text = []
        for i in range(len(para) // k + 1):
            if i < len(para) // k:
                batch = para[i * 1024:(i + 1) * 1024]
            else:
                batch = para[i * 1024:len(para)]

            # time_start = time.time()
            default_text.append([self.default_sum(2, 150, batch)])
            # time_end1 = time.time()
            # print('time cost1', time_end1 - time_start, 's')

            # time_start2 = time.time()
            # trained_text.append(self.pre_trained_sum(batch))
            # time_end2 = time.time()
            # print('time cost2', time_end2 - time_start2, 's')
        return default_text, trained_text

    """
    Tried using muti thread to improve the speed but it did not work:
    """
class myThread(threading.Thread):
    def __init__(self, threadID, name, counter, ifilename, ofilename):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.input_filename = ifilename
        self.output_filename = ofilename

    def run(self):
        print("start new thread： " + self.name)
        dt = DataTransfer()
        dt.transform_json(self.input_filename, self.output_filename)


if __name__ == "__main__":
    taker("subset_wapo_50k_sbert_ft_filtered.jl", "all-690-805.jl", 690)

    #Tried using muti thread to improve the speed
    # threads = []
    # for i in range(4):
    #     th = myThread(i+1, "Thread-%s" % str(i+1), i+1, "690-805-%s.jl" % i, "filted%s.json" % i)
    #     th.start()
    #     threads.append(th)
    #
    # for t in threads:
    #     t.join()
    # print("exit main thread")
    pass

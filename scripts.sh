"""
used in:
tfidf_weighting.py
sum_model_evaluate.py
get_sebert.py
"""
# load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions for TFIDF weighted embedding
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec

# load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3

"""
load index:
"""
# load wapo docs into the index called "wapo_docs_50k"
python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/subset_wapo_50k_sbert_ft_filtered.jl

# load test wapo docs using into the index called "test" and test it
python load_es_index.py --index_name test2 --wapo_path pa5_data/test.jl

# load test wapo docs using into the index called "test" and test it
python load_es_index.py --index_name test2 --wapo_path pa5_data/690-805-3-summarized.jl

"""
used in same as PA5:
tfidf_weighting.py
"""
# use title from topic 321 as the query; search over the custom_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type title -u --top_k 20

# use narration from topic 321 as the query; search over the content field from index "wapo_docs_50k" based on sentence BERT embedding and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type narration --vector_name sbert_vector --top_k 20

"""
used in same as PA5:
sum_model_evaluate.py
"""
# use narration from topic 321 as the query; search over the content field from index "wapo_docs_50k" based on sentence BERT embedding and compute NDCG@20
python sum_model_evaluate.py --index_name test2 --topic_id 321 --query_type narration --top_k 20
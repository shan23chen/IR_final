# load fasttext embeddings that are trained on wiki news. Each embedding has 300 dimensions
python -m embedding_service.server --embedding fasttext  --model pa5_data/wiki-news-300d-1M-subword.vec

# load sentence BERT embeddings that are trained on msmarco. Each embedding has 768 dimensions
python -m embedding_service.server --embedding sbert  --model msmarco-distilbert-base-v3

# load wapo docs into the index called "wapo_docs_50k"
python load_es_index.py --index_name wapo_docs_50k --wapo_path pa5_data/subset_wapo_50k_sbert_ft_filtered.jl
python load_es_index.py --index_name test2 --wapo_path pa5_data/test.jl


# use title from topic 321 as the query; search over the custom_content field from index "wapo_docs_50k" based on BM25 and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type title -u --top_k 20

# use narration from topic 321 as the query; search over the content field from index "wapo_docs_50k" based on sentence BERT embedding and compute NDCG@20
python evaluate.py --index_name wapo_docs_50k --topic_id 321 --query_type narration --vector_name sbert_vector --top_k 20
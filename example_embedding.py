from embedding_service.client import EmbeddingClient

if __name__ == "__main__":
    sbert_encoder = EmbeddingClient(
        host="localhost", embedding_type="sbert"
    )  # connect to the sbert embedding server
    fasttext_encoder = EmbeddingClient(
        host="localhost", embedding_type="fasttext"
    )  # connect to the fasttext embedding server
    texts = [
        "information retrieval is fun!",
        "Hello world!",
    ]  # encode two sentences/documents at the same time
    embedding = sbert_encoder.encode(texts)
    print(embedding.shape)  # shape is (2, 768)
    embedding = fasttext_encoder.encode(texts)
    print(embedding.shape)  # shape is (2, 300)

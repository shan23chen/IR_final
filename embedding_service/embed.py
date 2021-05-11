"""
wrapper for loading embeddings and encoding text
it will be called by the client
"""
from typing import List, Any
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

from embedding_service.text_processing import TextProcessing


class SBERTEmbedding:
    def __init__(self, model_name: str) -> None:
        """
        wrapper for loading sentence BERT embeddings (https://github.com/UKPLab/sentence-transformers)
        :param model_name: pretrained model name, check (https://www.sbert.net/docs/pretrained_models.html#) for other options
        """
        self.model = None
        self.load(model_name)

    def load(self, model_name: str) -> None:
        try:
            self.model = SentenceTransformer(model_name)
            print("Model loaded Successfully !")
        except Exception as e:
            print("Error loading Model, ", str(e))

    def encode(self, texts: List[str], pooling: Any = None) -> np.array:
        """
        encode a list of sentences into embeddings
        :param texts:
        :param pooling: no pooling method is needed for SBERT, argument passed in here is just a placeholder to make this method is consistent with fasttext
        :return:
        """
        try:
            assert self.model is not None
        except AssertionError:
            raise ValueError("model is not loaded!")
        text_embeddings = self.model.encode(texts, convert_to_numpy=True)
        return text_embeddings


class FastTextEmbedding:
    def __init__(self, model_path: str) -> None:
        """
        wrapper for loading fasttext embeddings (https://fasttext.cc/)
        :param model_path: local path to your downloaded embeddings in txt file from (https://fasttext.cc/docs/en/english-vectors.html)
        """
        self.word_vectors = {}
        self.unk_vector = np.zeros(300)  # default vector for unknown word
        self.load(model_path)
        self.text_processor = TextProcessing.from_nltk()

    def load(self, model_path: str) -> None:
        try:
            f = open(model_path, "r", encoding="utf-8")
            next(f)
            for line in tqdm(f):
                split_line = line.split()
                word = split_line[0]
                self.word_vectors[word] = np.array([float(val) for val in split_line[1:]])
            print("Model loaded Successfully !")
        except Exception as e:
            print("Error loading Model, ", str(e))

    def _single_encode_text(self, text: str, pooling: str = "mean") -> np.array:
        tokens = self._process_tokens(text)
        if not tokens:
            return self.unk_vector
        token_embeds = np.array(
            [self.word_vectors.get(token, self.unk_vector) for token in tokens]
        )
        if pooling == "mean":
            pooled = np.mean(token_embeds, axis=0)
            return pooled
        else:
            raise ValueError(f"cannot identify pooling method: {pooling}")

    def _process_tokens(self, text: str) -> List[str]:
        tokens = self.text_processor.get_valid_tokens("", text, use_stemmer=False)
        return tokens

    def encode(self, texts: List[str], pooling: str = "mean") -> np.array:
        """

        :param texts:
        :param pooling: default "mean", pooling method to reduce token embeddings into a single document embedding
        :return:
        """
        doc_embeddings = np.vstack(
            [self._single_encode_text(text, pooling) for text in texts]
        )
        return doc_embeddings


class Encoder:
    def __init__(self, embedding: str, model: str) -> None:
        """
        encoder wrapper for both type of embedding
        :param embedding: embedding types
        :param model: model name /path
        """
        self.embedding = embedding
        self.model = model
        self.embedding_model = None
        self._load()

    def _load(self) -> None:
        if self.embedding == "sbert":
            self.embedding_model = SBERTEmbedding(self.model)
        elif self.embedding == "fasttext":
            self.embedding_model = FastTextEmbedding(self.model)
        else:
            raise ValueError(f"cannot find model: {self.model}.")

    def encode(self, texts: List[str], pooling: str, batch_size: int = 256) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            vectors = self.embedding_model.encode(
                texts=texts[i : i + batch_size], pooling=pooling
            )
            embeddings.append(vectors)
        embeddings = np.vstack(embeddings)

        return embeddings


if __name__ == "__main__":
    pass

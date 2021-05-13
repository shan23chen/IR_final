#! /usr/bin/env python
"""
Query term expander.
Test by running this script.
Import to other module to use in IR system.
Author: Chester Palen-Michel
TODO add to a readme somewhere: nltk & wordnet, spaCy tokenizer, numpy
"""
from argparse import ArgumentParser
from enum import Enum
from operator import itemgetter
from typing import List, Optional, Sequence

from nltk.corpus import wordnet as wn, stopwords
from nltk.corpus.reader import Synset
from nltk.wsd import lesk
from numpy import dot
from numpy.linalg import norm
from spacy.lang.en import English

from embedding_service.client import EmbeddingClient


class WSD(Enum):
    FIRST = 1
    LESK = 2
    BERT = 3

    @staticmethod
    def string_to_enum(s: str) -> "WSD":
        if s == "FIRST":
            return WSD.FIRST
        elif s == "LESK":
            return WSD.LESK
        elif s == "BERT":
            return WSD.BERT
        else:
            raise ValueError("Unknown string for word sense enum")


class WordnetExpander:
    def __init__(self, wsd_strategy=WSD.FIRST, encoder=None, debug=False):
        self.stops = set(stopwords.words())
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.wsd_strategy = wsd_strategy
        self.encoder = encoder
        self.debug = debug
        self.embedding_cache = {}

    def check_valid_lemma(self, lemma: str):
        tokens = lemma.replace("_", " ").split()
        return all([len(t) > 2 for t in tokens])

    def expand(self, query_text: str) -> str:
        """
        Gets lemmas for the first synset for each non-stop word in
        a query text.
        """
        additional_tokens = []
        sent = [tok.text for tok in self.tokenizer(query_text)]
        sent_embedding = self.encoder.encode(sent, pooling="mean").tolist()[0]
        for token in self.tokenizer(query_text):
            if token.text in self.stops:
                continue
            syn = self.disambiguate_synset(
                self.wsd_strategy, token.text, sent, sent_embedding
            )
            if self.debug:
                lemma_names = syn.lemma_names() if syn else []
                print("SYNSET: ", syn, lemma_names)
            if syn is not None:
                additional_tokens.extend(
                    tuple(
                        lemma.name().replace("_", " ")
                        for lemma in syn.lemmas()
                        if self.check_valid_lemma(lemma.name())
                    )
                )
        return query_text + " " + " ".join(additional_tokens)

    def disambiguate_synset(
        self, strategy: WSD, token: str, sentence: List[str], sent_embedding
    ) -> Optional[Synset]:
        if strategy == WSD.FIRST:
            synsets = wn.synsets(token)
            if synsets:
                return synsets[0]
        elif strategy == WSD.LESK:
            return lesk(sentence, token)
        elif strategy == WSD.BERT:
            if self.encoder is None:
                raise ValueError(
                    "No encoder is specified can't use bert for disambiguation."
                )
            return self.embedding_disambiguate(sent_embedding, wn.synsets(token))
        return None

    def embedding_disambiguate(
        self, sent_embedding: Sequence[float], synsets: List[Synset]
    ) -> Optional[Synset]:
        """
        Find the synset that has highest cosine similarity with the sentence.
        Assume that this indicates a better match for the synsets meaning.
        Append the lemma names to the definition to get an embedding of the synset.
        """
        syn_scores = []
        if not synsets:
            return None
        for synset in synsets:
            faux_sent = tuple(
                [
                    t.text
                    for t in self.tokenizer(
                        " ".join(synset.lemma_names()) + synset.definition()
                    )
                ]
            )
            if faux_sent in self.embedding_cache:
                synset_embedding = self.embedding_cache[faux_sent]
            else:
                synset_embedding = self.encoder.encode(
                    list(faux_sent), pooling="mean"
                ).tolist()[0]
                self.embedding_cache[faux_sent] = synset_embedding
            cos_sim = dot(sent_embedding, synset_embedding) / (
                norm(sent_embedding) * norm(synset_embedding)
            )
            syn_scores.append((synset, cos_sim))
        # return the max scored synset
        return max(syn_scores, key=itemgetter(1))[0]


def test_expander() -> None:
    """
    Expands the query text to a new text and prints the results.
    For testing the expander on different queries.
    """
    parser = ArgumentParser()
    parser.add_argument("query", help="query string to expand")
    parser.add_argument(
        "--wsd",
        choices=["FIRST", "LESK", "BERT"],
        help="Strategy for word sense disambiguation.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    # Add a debug to print the different synsets
    encoder = EmbeddingClient(host="localhost", embedding_type="sbert")
    expander = WordnetExpander(
        wsd_strategy=WSD.string_to_enum(args.wsd), encoder=encoder, debug=args.debug
    )
    expansion = expander.expand(args.query)
    print(expansion)


if __name__ == "__main__":
    test_expander()

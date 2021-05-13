#! /usr/bin/env python
"""
Class and script to query wikidata and expand a query.
"""
import json
import re
import sys
import time
from multiprocessing.dummy import Pool
from typing import List

import spacy
from nltk.corpus import stopwords
import requests
from spacy.tokens.token import Token

QUERY_TERM_REPLACEMENT = "<QUERY_TERM>"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
WIKIDATA_QUERY = """SELECT DISTINCT ?subspecies ?subspeciesLabel ?subspeciesDescription WHERE {
  ?s ?label "<QUERY_TERM>"@en .
  ?s ?p ?o. 
  ?subspecies wdt:P31/wdt:P279* ?s . 
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
}
LIMIT 50"""


def valid_pos(t1: Token, t2: Token) -> bool:
    """
    Heuristic to get noun phrase looking things.
    In my trials, it's not common to get good hits from wikidata with > 2 tokens for
    these queries.
    """
    return (
        t1.pos_ in {"ADJ", "NOUN"}
        and t2.pos_ == "NOUN"
        and not {t1.text, t2.text}.intersection({"relevant", "document", "documents"})
    )


class WikidataExpander:
    def __init__(self):
        self.stops = set(stopwords.words())
        self.tokenizer = spacy.load("en_core_web_sm")

    def query_wikidata(self, query_texts: List[str]):
        """
        Forms a SPARQL query to send to wikidata. Then processes result dict
        to get the label and description for any matching entities.
        Filters out those with the wikidata identifier Q12345, etc.
        """

        ret = []
        for query_str in query_texts:
            # To avoid hitting the request limit
            time.sleep(2)
            query = re.sub(QUERY_TERM_REPLACEMENT, query_str, WIKIDATA_QUERY)
            r = requests.get(WIKIDATA_URL, params={"format": "json", "query": query})
            print("Status code: ", r.status_code)
            data = None
            try:
                data = r.json()
            except:
                print(f"{query_str} didn't work")
            if data is not None:
                for entry in data["results"]["bindings"]:
                    if "subspeciesLabel" in entry:
                        ret.append(entry["subspeciesLabel"]["value"])
                    if "subspeciesDescription" in entry:
                        ret.append(entry["subspeciesDescription"]["value"])
        return " ".join(
            token for token in ret if token and not re.match(r"Q\d+", token)
        )

    def expand(self, query_text: str) -> str:
        """
        Get additional terms using wikidata.
        """
        print(query_text)
        additional_terms = []
        tokens = [tok for tok in self.tokenizer(query_text)]
        bigrams = [
            (t1.text, t2.text)
            for t1, t2 in zip(tokens, tokens[1:])
            if valid_pos(t1, t2)
        ]
        # unigrams = [tok.text for tok in tokens if tok.pos_ == "NOUN"]
        additional_terms.append(
            self.query_wikidata([" ".join(bigram) for bigram in bigrams])
        )
        return query_text + " " + " ".join(additional_terms)


def wikidata_expand():
    query = sys.argv[1]
    expander = WikidataExpander()
    expanded_query = expander.expand(query)
    print(expanded_query)


if __name__ == "__main__":
    wikidata_expand()

import argparse
import time
from typing import List, Dict, Union, Iterator
from es_service.index import ESIndex
from utils import load_clean_wapo_with_embedding
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class IndexLoader:
    """
    load document index to Elasticsearch
    """

    def __init__(self, index, docs):

        self.index_name = index
        self.docs: Union[Iterator[Dict], List[Dict]] = docs

    def load(self) -> None:
        st = time.time()
        logger.info(f"Building index ...")
        ESIndex(self.index_name, self.docs)
        logger.info(
            f"=== Built {self.index_name} in {round(time.time() - st, 2)} seconds ==="
        )

    @classmethod
    def from_docs_jsonl(cls, index_name: str, docs_jsonl: str) -> "IndexLoader":
        try:
            return IndexLoader(index_name, load_clean_wapo_with_embedding(docs_jsonl))
        except FileNotFoundError:
            raise Exception(f"Cannot find {docs_jsonl}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_name",
        required=True,
        type=str,
        help="name of the ES index",
    )
    parser.add_argument(
        "--wapo_path",
        required=True,
        type=str,
        help="path to the processed wapo jsonline file",
    )

    args = parser.parse_args()
    idx_loader = IndexLoader.from_docs_jsonl(args.index_name, args.wapo_path)
    idx_loader.load()


if __name__ == "__main__":
    main()

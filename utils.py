from typing import Dict, Union, List, Generator
import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_clean_wapo_with_embedding(
    wapo_jl_path: Union[str, os.PathLike]
) -> Generator[Dict, None, None]:
    """
    load wapo docs as a generator
    :param wapo_jl_path:
    :return: yields each document as a dict
    """
    with open(wapo_jl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield json.loads(line)


def parse_wapo_topics(xml_file: str) -> Dict[str, List[str]]:
    """
    parse topics2018.xml
    :param xml_file:
    :return: a dict that maps the topic id to its title. narrative and description
    """
    text = open(xml_file, "r").read()
    topic_mapping = defaultdict(list)

    for xml_str in text.strip().split("\n\n"):
        tree = ET.fromstring(xml_str)
        topic_id = ""
        for child in tree:
            if child.text:
                if child.tag == "num":
                    # get topic id
                    topic_id = child.text.split(":")[-1].strip()
                else:
                    # append others to this topic id
                    topic_mapping[topic_id].append(child.text.strip().split("\n")[-1])
    return topic_mapping


if __name__ == "__main__":
    pass

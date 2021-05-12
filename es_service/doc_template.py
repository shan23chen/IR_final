from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import (  # type: ignore
    Document,
    Text,
    Keyword,
    Date,
    analyzer,
)

# TODO: build your own analyzer
custom_analyzer = analyzer(
    'custom_analyzer',
    tokenizer='standard',
    filter=['lowercase', 'asciifolding', 'snowball']
)


class BaseDoc(Document):
    """
    wapo document mapping structure
    """

    doc_id = (
        Keyword()
    )  # we want to treat the doc_id as a Keyword (its value won't be tokenized or normalized).
    title = (
        Text()
    )  # by default, Text field will be applied a standard analyzer at both index and search time
    author = Text()
    content = Text(
        analyzer="standard"
    )  # we can also set the standard analyzer explicitly
    # custom_content = Text(
    #     analyzer=custom_analyzer
    # )  # TODO: uncomment this to index the same content again with your custom analyzer
    date = Date(
        format="yyyy/MM/dd"
    )  # Date field can be searched by special queries such as a range query.
    annotation = Text()
    # ft_vector = DenseVector(dims=300)  # fasttext embedding in the DenseVector field
    # sbert_vector = DenseVector(dims=768)  # sentence BERT embedding in the DenseVector field
    sbert_vector = Text()   # todo

    def save(self, *args, **kwargs):
        """
        save an instance of this document mapping in the index
        this function is not called because we are doing bulk insertion to the index in the index.py
        """
        return super(BaseDoc, self).save(*args, **kwargs)


if __name__ == "__main__":
    pass

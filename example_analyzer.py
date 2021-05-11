from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import analyzer, tokenizer, token_filter, char_filter

if __name__ == "__main__":
    connections.create_connection(hosts=["localhost"], timeout=100, alias="default")

    # customize an analyzer by specifying a tokenizer and filters.
    my_analyzer1 = analyzer(
        "my_analyzer1",
        tokenizer=tokenizer("trigram", "ngram", min_gram=3, max_gram=3),
        filter=["lowercase"],
    )
    response = my_analyzer1.simulate(
        "The big fox jumps over the lazy dog!"
    )  # simulate the analyzer effect on text

    tokens = [t.token for t in response.tokens]
    print(tokens)

    my_analyzer2 = analyzer(
        "my_analyzer2",
        tokenizer="standard",
        filter=["asciifolding"],
    )
    response = my_analyzer2.simulate("açaí à HA la carte")

    tokens = [t.token for t in response.tokens]
    print(tokens)

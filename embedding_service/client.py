"""
embedding client
adapted from https://github.com/amansrivastava17/embedding-as-service
"""

from typing import Union, List, Optional
import numpy as np
import zmq
import json

from embedding_service import INV_PORT_EMBEDDING_MAPPING


class EmbeddingClient(object):
    """
    Represents an example client.
    """

    def __init__(self, host, embedding_type):
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{host}:{INV_PORT_EMBEDDING_MAPPING[embedding_type]}")
        self.identity = "123"

    def encode(
        self,
        texts: Union[List[str], List[List[str]]],
        pooling: Optional[str] = "mean",
        batch_size: int = 256,
        **kwargs,
    ) -> np.array:
        """
        Connects to server. Send compute request, poll for and print result to standard out.
        """
        if not isinstance(texts, list):
            raise ValueError(
                "Argument `texts` should be either List[str] or List[List[str]]"
            )
        embeddings = []
        for i in range(0, len(texts), batch_size):
            request_data = {
                "type": "encode",
                "texts": texts[i : i + batch_size],
                "pooling": pooling,
            }
            self.send(json.dumps(request_data))
            result = self.receive()
            result = json.loads(result.decode("utf-8"))
            embeddings.append(np.array(result))
        embeddings = np.vstack(embeddings)
        return embeddings

    def terminate(self):
        self.socket.close()
        self.zmq_context.term()

    def send(self, data):
        """
        Send data through provided socket.
        """
        self.socket.send_string(data)

    def receive(self):
        """
        Receive and return data through provided socket.
        """
        return self.socket.recv()

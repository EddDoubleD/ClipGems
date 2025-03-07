import os

from pymilvus import MilvusClient


class GemRepositoryConfig(object):

    def __init__(self):
        self.url = os.getenv('url', 'http://localhost:19530')
        self.token = os.getenv('token', 'root:Milvus')


class GemRepository(object):

    def __init__(self, config: GemRepositoryConfig, collection: str, vector_field: str = "embedding"):
        self.client = MilvusClient(uri=config.url, token=config.token)
        self.collection = collection
        self.vector_field = vector_field

    def index(self, data):
        self.client.insert(
            collection_name=self.collection,
            data=data
        )

    def search(self, data, limit=5, out=None):
        if out is None:
            out = ["pk"]

        return self.client.search(
            collection_name=self.collection,
            data=[data],
            anns_field=self.vector_field,
            params={"metric_type": "IP"},
            limit=limit,
            output_fields=out
        )
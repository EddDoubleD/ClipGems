import os

from pymilvus import MilvusClient


class GemRepositoryConfig(object):

    def __init__(self):
        self.url = os.getenv('MILVUS_URL', 'http://localhost:19530')
        self.token = os.getenv('MILVUS_TOKEN', 'root:Milvus')


class GemRepository(object):

    def __init__(self, config: GemRepositoryConfig, collection: str, vector_field: str = "embedding"):
        self.client = MilvusClient(uri=config.url, token=config.token)
        self.collection = collection
        self.vector_field = vector_field

    def search(self, data, limit=5, out=None):
        if out is None:
            out = ["pk"]

        return self.client.search(
            collection_name=self.collection,
            data=[data],
            anns_field=self.vector_field,
            params={"metric_type": "COSINE"},
            limit=limit,
            output_fields=out
        )

import os

from pymilvus import MilvusClient


class GemRepository(object):

    def __init__(self, collection: str, vector_field: str = "embedding"):
        self.url = os.getenv('MILVUS_URL', 'http://localhost:19530')
        self.token = os.getenv('MILVUS_TOKEN', 'root:Milvus')

        self.client = MilvusClient(
            uri=self.url,
            token=self.token
        )

        self.collection = collection
        self.vector_field = vector_field

    def index(self, data):
        self.client.insert(
            collection_name=self.collection,
            data=data
        )

    def delete(self, ids: list[str]):
        deleted = self.client.delete(
            collection_name=self.collection,
            ids=ids
        )

        return deleted
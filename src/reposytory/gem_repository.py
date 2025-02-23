from pymilvus import MilvusClient


class GemRepository(object):

    def __init__(self, collection: str, vector_field: str = "embedding"):
        self.client = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus"
        )

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
            params={"nprobe": 10},
            limit=limit,
            output_fields=out
        )

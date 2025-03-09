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

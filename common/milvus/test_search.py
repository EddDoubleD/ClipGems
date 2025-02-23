from pymilvus import MilvusClient, DataType

if __name__ == '__main__':
    try:
        client = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus"
        )

        search_params = {
            "params": {"nprobe": 10}
        }
        query_vector = [0.1, 0.2, 0.3, 0.7]

        res = client.search(
            collection_name="test",
            data=[query_vector],
            anns_field="embedding",
            search_params=search_params,
            limit=5,
            output_fields=["pk"]
        )

        for x in res[0]:
            print(str(x))
            res = client.delete(
                collection_name="test",
                ids=[x["pk"]]
            )

    except Exception as e:
        print(f"Connect to Milvus collection failed: {e}")
        exit(1)
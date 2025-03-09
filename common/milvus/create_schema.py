from pymilvus import MilvusClient, DataType

if __name__ == '__main__':
    try:
        client = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus"
        )

        # create schema
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        # Add fields to schema
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        # automatically decides the most appropriate index type
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        # create collection
        client.create_collection(
            collection_name="gems",
            schema=schema,
            index_params=index_params
        )
        print("Milvus client created successfully")
    except Exception as e:
        print(f"Create Milvus client failed: {e}")
        exit(1)

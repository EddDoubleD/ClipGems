from pymilvus import MilvusClient, DataType

if __name__ == '__main__':
    try:
        client = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus"
        )

        # Data to be inserted
        data = [
            {
                "pk": "3",
                "metadata": {"category": "electronics", "price": 99.99, "brand": "BrandA"},
                "embedding": [0.1200, 0.3400, 0.5600, 0.9100],
                "bucket": "null"
            },
            {
                "pk": "2",
                "metadata": {"category": "home_appliances", "price": 249.99, "brand": "BrandB"},
                "embedding": [0.5600, 0.7800, 0.9000, 0.9100],
                "bucket": "null"
            },
            {
                "pk": "1",
                "metadata": {"category": "furniture", "price": 399.99, "brand": "BrandC"},
                "embedding": [0.9100, 0.1800, 0.2300, 0.9100],
                "bucket": "null"
            }
        ]

        # Insert data into the collection
        client.insert(
            collection_name="test",
            data=data
        )
        print("Data inserted successfully")
    except Exception as e:
        print(f"Connect to Milvus client failed: {e}")
        exit(1)
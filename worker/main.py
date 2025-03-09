import asyncio
import logging

from predictor import Predictor
from gem_finder import GemFinder
from storage.milvus import GemRepository
from storage.s3 import S3ClientFactory

logging.basicConfig(
    level=logging.INFO,
    filename="data/error.log",
    filemode="w+",
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    try:
        predictor = Predictor()
        predictor.load()
        s3_client_factory = S3ClientFactory()
        repository: GemRepository = GemRepository(
            collection="gems"
        )
        consumer = GemFinder(
            predictor=predictor,
            repository=repository,
            s3_client_factory=s3_client_factory
        )
        asyncio.run(consumer.start())
    except Exception as e:
        logger.error(f"Failed to start consumer: {e}")
        exit(1)

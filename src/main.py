import asyncio
import logging
import os
import threading
from typing import Dict
from uuid import uuid4

import clip
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException

from model.dto import ProcessRequest, JobStatus
from reposytory.file_storage import S3ClientFactory
from reposytory.gem_repository import GemRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

try:
    S3_CLIENT_FACTORY = S3ClientFactory()
except Exception as e:
    logger.error(f"Failed to create S3 client factory: {e}")
    exit(1)
try:
    repo: GemRepository = GemRepository(
        collection="gems"
    )
except Exception as e:
    logger.error(f"Failed to create GemRepository: {e}")
    exit(1)

app = FastAPI()

jobs: Dict[str, Dict] = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

lock = threading.Lock()


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/v1/status")
async def job_status(job: str):
    if job not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": jobs[job]["status"]}


@app.get("/api/v1/stat")
async def job_stat():
    return {"jobs": jobs}


@app.post("/api/v1/photo/process")
async def process_photo(request: ProcessRequest):
    """
    
    :param request: ProcessRequest 
    :return: 200 if file exists and create task
    """
    lock.acquire()
    failed = False
    try:
        job_id = str(uuid4())
        jobs[job_id] = {
            "status": JobStatus.PENDING
        }
        asyncio.create_task(async_process_photo(job_id, request.bucket, request.path))
        return {"job_id": job_id}
    except Exception as e:
        logger.info("Err: " + str(e))
        failed = True
    finally:
        # Always called, even if exception is raised in try block
        lock.release()
        if failed:
            raise HTTPException(status_code=500, detail="cannot create job")


@app.get("/api/v1/search")
async def emb(text: str):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(text_input)
        vector = normalize_vector(features)
        float_embedding = list()
        for x in vector.tolist()[0]:
            float_embedding.append(float(x))

        try:
            response = repo.search(float_embedding, out=["bucket", "metadata"])
            return {"hits": str(response)}
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail="Search error")


def normalize_vector(vector):
    """
    Takes a vector and returns its normalized version

    :param vector: original vector of arbitrary length
    :return: vector of the same direction, with length 1
    """
    return vector / torch.linalg.norm(vector)


def create_path(dir_path: str, file_name: str = None):
    if file_name is None:
        return f"./{dir_path}"

    return f"{dir_path}/{file_name}"


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)


def get_image_features(vector):
    with torch.no_grad():
        image_features = model.encode_image(vector)
    return image_features


async def async_process_photo(job_id: str, bucket: str, file_path: str):
    logger.info(f"Start processing job {job_id}")
    jobs[job_id]["status"] = JobStatus.WORK
    try:
        # check lock job by id
        local_dir_path = create_path(job_id)
        os.mkdir(local_dir_path, mode=0o777)
        s3_client = S3_CLIENT_FACTORY.client()

        file_name = file_path.split("/")[-1]
        local_file_path = create_path(local_dir_path, file_name)
        s3_client.download_file(Bucket=bucket, Key=file_path, Filename=local_file_path)

        logger.info(f"Success download file: {file_name} to {local_file_path}")
        image_vector = load_and_preprocess_image(local_file_path)
        features = get_image_features(image_vector)
        vector = normalize_vector(features)
        vector_list = vector.tolist()
        embedding = ", ".join(f"{x:.32}" for x in vector_list[0])
        jobs[job_id] = {
            "status": JobStatus.DONE,
            "embedding": embedding
        }

        float_embedding = []
        for x in vector_list[0]:
            float_embedding.append(float(x))

        repo.index(
            [
                {
                    "pk": f"{bucket}/{file_path}",
                    "metadata": {"job_id": job_id, "file_path": file_path},
                    "embedding": float_embedding,
                    "bucket": bucket
                }
            ]
        )
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
    finally:
        local_dir_path = create_path(job_id)
        for root, dirs, files in os.walk(local_dir_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        if os.path.exists(local_dir_path):
            os.rmdir(local_dir_path)
        logger.info(f"End processing job {job_id}")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)

import asyncio
import logging
import os
import threading
from typing import Dict
from uuid import uuid4

import clip
import cv2
import numpy as np
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from scenedetect import SceneManager, open_video, StatsManager
from scenedetect.detectors import ContentDetector

from vectorazzi.src.vectorazzi.dto import ProcessRequest, JobStatus
from vectorazzi.src.vectorazzi.file_storage import S3ClientFactory
from vectorazzi.src.vectorazzi.gem_repository import GemRepository

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


@app.get("/healthcheck")
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


@app.post("/api/v1/photo")
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


@app.post("/api/v1/clip")
async def create_clip(request: ProcessRequest):
    job_id = str(uuid4())

    # Создание новой задачи
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "stat": "",
    }

    # Запуск обработки видео в фоне
    asyncio.create_task(process_video(job_id, request.bucket, request.path))

    return {"job_id": job_id}


@app.get("/api/v1/search")
async def search(text: str):
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(text_input)
        vector = normalize_vector(features)
        float_embedding = list()
        for x in vector.tolist()[0]:
            float_embedding.append(float(x))

        try:
            response = repo.search(float_embedding, out=["bucket", "metadata"])
            return response[0]
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
        return f"./data/{dir_path}"

    return f"{dir_path}/{file_name}"


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)


def get_image_features(image):
    with torch.no_grad():
        image_features = model.encode_image(image)
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


async def process_video(job_id: str, bucket: str, file_path: str):
    file_name = file_path.split("/")[-1]
    try:
        local_dir_path = create_path(job_id)
        out = f"{local_dir_path}/report.csv"
        os.mkdir(local_dir_path, mode=0o777)
        s3 = S3_CLIENT_FACTORY.client()
        # Обновление статуса задачи
        jobs[job_id]["status"] = JobStatus.WORK
        local_file_path = f"{local_dir_path}/{file_name}"
        # Скачивание видео
        s3.download_file(Bucket=bucket, Key=file_path, Filename=local_file_path)

        try:
            video = open_video(local_file_path)
            scene_manager = split_video_into_scenes()
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list(start_in_scene=True)
            print('List of scenes obtained:')

            cap = cv2.VideoCapture(local_file_path)
            fps = round(cap.get(cv2.CAP_PROP_FPS))

            for i, scene in enumerate(scene_list):
                vectors = []
                logger.info(f"Scene {i + 1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")
                os.mkdir(f'{local_dir_path}/{i + 1}')
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                while cap.isOpened() and start_frame <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Если текущий кадр находится в нужном интервале, сохраняем его
                    current_frame_number = int(cap.get(1)) - 1
                    if current_frame_number % fps == 0 and start_frame <= current_frame_number <= end_frame:

                        image = Image.fromarray(frame)
                        image_vector = preprocess(image).unsqueeze(0).to(device)
                        feature = get_image_features(image_vector)
                        vectors.append(feature)
                        # debug options
                        # output_frame_path = f'frame_{current_frame_number}.jpg'
                        # cv2.imwrite(f'{local_dir_path}/{i + 1}/{output_frame_path}', frame)

                    start_frame += 1  # Переходим к следующему кадру в интервале

                avg_vector = np.average(vectors, 0)
                tensor = torch.linalg.Tensor(avg_vector)
                vector = normalize_vector(tensor)
                vector_list = vector.tolist()
                float_embedding = []
                for x in vector_list[0]:
                    float_embedding.append(float(x))

                repo.index(
                    [
                        {
                            "pk": f"{bucket}/{file_path}_{i + 1}",
                            "metadata": {"job_id": job_id, "file_path": file_path, "from": scene[0].get_seconds(), "to": scene[1].get_seconds()},
                            "embedding": float_embedding,
                            "bucket": bucket
                        }
                    ]
                )
            cap.release()
            # scene_manager.stats_manager.save_to_csv(csv_file=out)
        except Exception as ex:
            logger.error(ex)
        finally:
            # Обновление статуса задачи
            jobs[job_id]["status"] = JobStatus.DONE
    except Exception as ex:
        logger.error(f"Error processing video: {ex}")
    finally:
        jobs[job_id]["status"] = JobStatus.DONE


def split_video_into_scenes(threshold=27.0) -> SceneManager:
    """
    Open our video, create a scene manager, and add a detector.
    :param threshold:
    :return:
    """

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    return scene_manager


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)

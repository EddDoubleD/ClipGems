import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict
import boto3
import asyncio
from enum import Enum
import os
from scenedetect import detect, VideoManager, SceneManager, open_video, split_video_ffmpeg
from scenedetect.detectors import ContentDetector

app = FastAPI()

AWS_KEY_ID=os.getenv('AWS_KEY_ID')
AWS_KEY=os.getenv('AWS_KEY')

class JobStatus(str, Enum):
    PENDING = 'PENDING'
    WORK = 'WORK'
    DONE = 'DONE'


class ClipRequest(BaseModel):
    bucket: str
    path: str


jobs: Dict[str, Dict] = {}
session = boto3.session.Session()


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/clip")
async def create_clip(request: ClipRequest):
    job_id = str(uuid4())
    result_bucket = f"{request.bucket}-results"

    # Создание нового бакета для результатов
    # s3.create_bucket(Bucket=result_bucket)

    # Создание новой задачи
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "result_bucket": result_bucket,
    }

    # Запуск обработки видео в фоне
    asyncio.create_task(process_video(job_id, request.bucket, request.path, result_bucket))

    return {"job_id": job_id, "result_bucket": result_bucket}


@app.get("/api/v1/status")
async def job_status(job: str):
    if job not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": jobs[job]["status"]}


async def process_video(job_id: str, bucket: str, path: str, result_bucket: str):
    file_name = path.split("/")[-1]
    local_path = f"/tmp/{file_name}"

    try:

        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=AWS_KEY_ID,
            aws_secret_access_key=AWS_KEY
        )
        # Обновление статуса задачи
        jobs[job_id]["status"] = JobStatus.WORK


        # Скачивание видео
        s3.download_file(Bucket=bucket, Key=path, Filename=local_path)

        # Analyzing video with PySceneDetect
        scene_list = split_video_into_scenes(local_path)
        for i, scene in enumerate(scene_list):
            print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i + 1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))

    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Обновление статуса задачи
        jobs[job_id]["status"] = JobStatus.DONE
        # Удаление локального файла
        if os.path.exists(local_path):
            os.remove(local_path)

def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    return scene_manager.get_scene_list(start_in_scene=True)


if __name__ == '__main__':

    uvicorn.run(app, host="0.0.0.0", port=8000)
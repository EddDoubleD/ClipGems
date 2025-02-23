import uvicorn
from fastapi import FastAPI, HTTPException
from uuid import uuid4
from typing import Dict
import boto3
import asyncio
import os
from scenedetect import SceneManager, open_video, StatsManager
from scenedetect.detectors import ContentDetector

from src.model.dto import ProcessRequest, JobStatus

app = FastAPI()

AWS_KEY_ID = os.getenv('AWS_KEY_ID')
AWS_KEY = os.getenv('AWS_KEY')


jobs: Dict[str, Dict] = {}
session = boto3.session.Session()


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/clip")
async def create_clip(request: ProcessRequest):
    job_id = str(uuid4())

    # Создание нового бакета для результатов
    # s3.create_bucket(Bucket=result_bucket)

    # Создание новой задачи
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "stat": "",
    }

    # Запуск обработки видео в фоне
    asyncio.create_task(process_video(job_id, request.bucket, request.path))

    return {"job_id": job_id}


@app.get("/api/v1/status")
async def job_status(job: str):
    if job not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": jobs[job]["status"]}


async def process_video(job_id: str, bucket: str, path: str):
    file_name = path.split("/")[-1]
    local_path = f"/tmp/{file_name}"
    out = f"/tmp/{job_id}.csv"
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

        try:
            video = open_video(local_path)
            scene_manager = split_video_into_scenes()
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list(start_in_scene=True)
            print('List of scenes obtained:')
            for i, scene in enumerate(scene_list):
                print(
                    'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                        i + 1,
                        scene[0].get_timecode(), scene[0].get_frames(),
                        scene[1].get_timecode(), scene[1].get_frames(),)
                )

            scene_manager.stats_manager.save_to_csv(csv_file=out)
        finally:
            # Обновление статуса задачи
            jobs[job_id]["status"] = JobStatus.DONE
            jobs[job_id]["stat"] = out
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        jobs[job_id]["status"] = JobStatus.DONE
        jobs[job_id]["stat"] = out


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
    uvicorn.run(app, host="0.0.0.0", port=8000)

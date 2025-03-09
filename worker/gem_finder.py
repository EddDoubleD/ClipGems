import json
import logging
import os
import threading
from typing import List

import cv2
from PIL import Image
from storage.milvus import GemRepository
from scenedetect import open_video

from messaging.consumer import Consumer
from messaging.message import Message
from storage.s3 import S3ClientFactory
from worker.dto import Event
from worker.predictor import Predictor
from worker.utils import create_path, split_video_into_scenes, extract_frames

logging.basicConfig(
    level=logging.INFO,
    filename="data/app.log",
    filemode="w+",
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger("gem_finder")


class GemFinder(Consumer):

    def __init__(
            self,
            predictor: Predictor,
            repository: GemRepository,
            s3_client_factory: S3ClientFactory
    ):
        super().__init__()
        self.predictor = predictor
        self.repository = repository
        self.s3_client_factory = s3_client_factory

    def handle_message_batch(self, messages: List[Message]):
        if messages.__len__() == 0:
            logger.info("hmb batch is empty")
            return
        tasks = []
        logger.info(f"hmb batch: {messages.__len__()}")
        for i, message in enumerate(messages):
            _json = json.loads(message.Body.replace("\'", "\""))
            event = Event.parse(
                _json['messages'][0]
            )

            if event.type == 'yandex.cloud.events.storage.ObjectDelete':
                task = threading.Thread(target=self.delete_file_from_repo, args=(event,))
                tasks.append(task)
                task.start()  # Запускаем поток
                continue

            if event.object_id.endswith('png') or event.object_id.endswith('jpg'):
                task = threading.Thread(target=self.image_process, args=(event,))
                tasks.append(task)
                task.start()  # Запускаем поток
                continue

            if event.object_id.endswith('mkv') or event.object_id.endswith('mp4') or event.object_id.endswith('mpeg4'):
                task = threading.Thread(target=self.video_process, args=(event,))
                tasks.append(task)
                task.start()  # Запускаем поток
                continue

        for task in tasks:
            task.join()

    def handle_batch_processing_exception(self, messages: List[Message], exception):
        logger.error(f"Handling error {str(exception)}")

    def delete_file_from_repo(self, event: Event):
        # stubbed
        # deleted_count = 1
        deleted_count = self.repository.delete(list(f"{event.bucket_id}/{event.object_id}"))
        if deleted_count:
            logger.info(f"Object {event.bucket_id}/{event.object_id} successfully deleted")
        else:
            logger.info(f"Object {event.bucket_id}/{event.object_id} not found in repo")

    def image_process(self, event: Event):
        """

        :param event:
        :return:
        """
        local_dir_path = create_path(event.id)
        try:
            os.mkdir(local_dir_path, mode=0o777)
            s3_client = self.s3_client_factory.client()
            file_name = event.object_id.split("/")[-1]
            local_file_path = create_path(local_dir_path, file_name)
            s3_client.download_file(Bucket=event.bucket_id, Key=event.object_id, Filename=local_file_path)
            logger.info(f"Success download file: {file_name} to {local_file_path}")
        except Exception as ex:
            logger.error(f"Error during download file: {ex}")
            return

        try:
            image = Image.open(local_file_path)
            features = self.predictor.mean_image(image)
            embedding = [float(x) for x in features.numpy()[0]]
            self.repository.index(
                [
                    {
                        "pk": f"{event.bucket_id}/{event.object_id}",
                        "metadata": {
                            "job_id": event.id,
                            "original": f"{event.bucket_id}/{event.object_id}",
                            "type": "image"
                        },
                        "embedding": embedding
                    }
                ]
            )
        except Exception as ex:
            logger.error(f"Error during image processing: {ex}")
        finally:
            for root, dirs, files in os.walk(local_dir_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            if os.path.exists(local_dir_path):
                os.rmdir(local_dir_path)
            logger.info(f"End processing job {event.id}")

    def video_process(self, event: Event):
        """

        :param event:
        :return:
        """
        local_dir_path = create_path(event.id)
        file_name = event.object_id.split("/")[-1]
        try:
            os.mkdir(local_dir_path, mode=0o777)
            s3_client = self.s3_client_factory.client()
            local_file_path = create_path(local_dir_path, file_name)
            s3_client.download_file(Bucket=event.bucket_id, Key=event.object_id, Filename=local_file_path)
            logger.info(f"Success download file: {file_name} to {local_file_path}")
        except Exception as e:
            logger.error(f"Error during download file: {e}")
            return

        try:
            video = open_video(local_file_path)
            scene_manager = split_video_into_scenes()
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list(start_in_scene=True)
            cap = cv2.VideoCapture(local_file_path)
            for i, scene in enumerate(scene_list):
                logger.info(
                    f"{event.id}: Scene {i + 1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")
                local_fragment_dir_path = f'{local_dir_path}/{i + 1}'
                os.mkdir(local_fragment_dir_path)
                frames = [Image.fromarray(frame) for frame in extract_frames(capture=cap, scene=scene)]
                features = self.predictor.mean_images(frames)
                embedding = [float(x) for x in features[0]]
                logger.info(f"{event.id} success in ms {embedding}")
                preset_file_path = f'{local_fragment_dir_path}/{file_name}_{i}.png'
                frames[0].save(preset_file_path)
                s3_client.upload_file(
                    preset_file_path,
                    'clip-gems-preset',
                    f'{file_name}_{i}.png',
                    ExtraArgs={"ChecksumSHA256": "UNSIGNED-PAYLOAD"}
                )
                self.repository.index(
                    [
                        {
                            "pk": f"clip-gems-preset/{file_name}_{i}.png",
                            "metadata": {
                                "job_id": event.id,
                                "original": f"{event.bucket_id}/{event.object_id}",
                                "type": "video"
                            },
                            "embedding": embedding
                        }
                    ]
                )
        except Exception as ex:
            logger.error(f"Error during video processing: {ex}")
        finally:
            for root, dirs, files in os.walk(local_dir_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            if os.path.exists(local_dir_path):
                os.rmdir(local_dir_path)
            logger.info(f"End processing job {event.id}")

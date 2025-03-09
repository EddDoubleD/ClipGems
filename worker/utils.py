import cv2
from typing import Tuple
from scenedetect import StatsManager, SceneManager, ContentDetector, FrameTimecode


def create_path(dir_path: str, file_name: str = None):
    if file_name is None:
        return f"./data/{dir_path}"

    return f"{dir_path}/{file_name}"


def split_video_into_scenes(threshold=27.0) -> SceneManager:
    """
    Open our video, create a scene manager, and add a detector.
    """

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    return scene_manager


def extract_frames(capture: cv2.VideoCapture, scene: Tuple[FrameTimecode, FrameTimecode]):
        """

        :param capture: openCV video capture
        :param scene: detected scene
        :return: empedding of the scene
        """

        fps = round(capture.get(cv2.CAP_PROP_FPS))
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        frames = []
        while capture.isOpened() and start_frame <= end_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = capture.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            start_frame += fps

        return frames

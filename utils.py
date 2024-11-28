import os
import cv2
import tempfile
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.scene_manager import save_images


def extract_frames(video_path, output_dir, force_uniform=False):
    """
    Extracts frames from a video and saves them to the output directory.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        str: list of image paths of the extracted frames.
    """
    # Create the directories to save keyframes
    os.makedirs(output_dir, exist_ok=True)

    # Initialize VideoManager, StatsManager, and SceneManager
    video_manager = VideoManager([video_path])

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    # Add a content detector to the SceneManager
    scene_manager.add_detector(ContentDetector())

    # Start video processing
    video_manager.set_downscale_factor()
    video_manager.start()

    # Detect scenes
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    # If no scenes detected, save frames at regular intervals
    if not scene_list or force_uniform:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // 10)  # Save ~10 frames
        frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frame_path = os.path.join(output_dir, f"uniform_frame_{saved_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            frame_idx += 1
        cap.release()
    else:
        # Save keyframes for each detected scene
        save_images(scene_list, video_manager, num_images=1, output_dir=output_dir)

    video_manager.release()

    return [os.path.join(output_dir, each_video_dir) for each_video_dir in os.listdir(output_dir)]
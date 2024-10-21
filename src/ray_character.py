import torch
from ray import serve
import os
from .live_portrait_character import LivePortraitCharacter
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.video import images2video
import numpy as np
import cv2
from .rem_bg.rembg_class import RembgWrapper
from contextlib import contextmanager

@contextmanager
def to_cv2(data, path):
    try:
        cv2.imwrite(path, data, [int(cv2.IMWRITE_WEBP_QUALITY), 20])
        with open(path, "rb") as fr:
            d = fr.read()
        yield d
    finally:
        if os.path.isfile(path):
            os.remove(path)

# _preset_motion = [
#     9, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
#     10, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
#     11, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437,
#     14, 437, 436, 435, 434, 433, 432, 431, 430, 429, 428,
#     11, 407, 406, 405, 404, 403, 402, 401, 400, 399, 398,
#     10, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128,
#     1, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
#     2, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68,
#     1, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
#     6, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108,
#     1, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
#     10, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
#     11, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168,
#     2, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 0, 28, 29, 30
# ]
_preset_motion = [
    398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    11, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437,
    14, 437, 436, 435, 434, 433, 432, 431, 430, 429, 428,
    11, 407, 406, 405, 404, 403, 402, 401, 400, 399, 398,
    10, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128,
    1, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    2, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68,
    1, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    6, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108,
    1, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
    10, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    11, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168,
    2, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 0, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 2, 37, 36, 35, 34
]
# _preset_motion = [
#     9, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
#     10, 407, 406, 405, 404, 403, 402, 401, 400, 399, 398,
#     9, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
#     10, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447,
#     11, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188,
#     2, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
#     5, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158,
#     2, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
#     4, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148,
#     2, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
#     11, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487,
#     14, 337, 336, 335, 334, 333, 332, 331, 330, 329, 328,
#     5, 257, 256, 255, 254, 253, 252, 251, 250, 249, 248,
#     4, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
#     6, 267, 266, 265
# ]
_preset_motion = [
    407, 406, 405, 404, 403, 402, 401, 400, 399, 398,
    9, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    10, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447,
    11, 447, 446, 445, 444, 443, 443, 444, 445, 446, 447,
    11, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188,
    2, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    5, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158,
    2, 148, 149, 150, 151, 152, 152, 151, 150, 149, 148,
    2, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    11, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487,
    14, 487, 486, 485, 484, 483, 482, 481, 480, 479, 478,
    11, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188,
    2, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    5, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158,
    2, 148, 149, 150,
]
_preset_motion_unique = list(set(_preset_motion))


os.environ["RAY_STAGE"] = "local"
ray_stage = os.environ["RAY_STAGE"]

serve.start(http_options={"host": "0.0.0.0", "port": 8001})

@serve.deployment(
    health_check_timeout_s=60,
    health_check_period_s=90,
    max_concurrent_queries=24,
    autoscaling_config={
        "min_replicas": 0 if ray_stage != "release" else 2,
        "max_replicas": 10 if ray_stage != "release" else 20,
        "target_num_ongoing_requests_per_replica": 18 if ray_stage != "release" else 18,
        "initial_replicas": 1 if ray_stage != "release" else 20,
        "metrics_interval_s": 1.,
        "look_back_period_s": 60. if ray_stage != "release" else 30.0,
        "downscale_smoothing_factor": 0.5 if ray_stage != "release" else 0.2,
        "upscale_smoothing_factor": 1. if ray_stage != "release" else 0.7,
        "downscale_delay_s": 1800 if ray_stage != "release" else 120,
        "upscale_delay_s": 0.0 if ray_stage != "release" else 0.0,
    },
    ray_actor_options={"num_cpus": 1, "num_gpus": 1, "runtime_env": {"conda": "foo"}},
)
class CharacterHandler(LivePortraitCharacter):
    def __init__(self):
        super().__init__(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())
        self.bg_remover = RembgWrapper()

    @serve.batch(max_batch_size=1, batch_wait_timeout_s=1)
    async def __call__(self, request):
        from time import time
        request = request[0]
        json_data = await request.json()

        ###
        with open("img_temp.jpg", "wb") as fw:
            fw.write(eval(json_data["img"]))
        img = cv2.imread("img_temp.jpg")
        s = time()
        # img = self.bg_remover.remove(img)
        # print(f"{time()-s:.3f}")
        ###
        # img = torch.tensor(json_data["img"]).numpy().astype(np.uint8)
        eye = self.get_initial_eye(img)
        uid = json_data.get("uid", "temp")
        preview = json_data.get("preview", False)
        trajectory = torch.load("preset/demo_webp/trajectory.pkl")
        landmarks = trajectory.graph["landmarks"]
        frames = []
        print(f"Receive request, start generating {'preview' if preview else 'full-source'}")
        s = time()
        lmk, _ = self.cropper.fa.get_landmarks(img)
        lmk = lmk[0]
        if preview:
            preview_clip = self.generate_preview(landmarks, img, uid, eye, lmk)
        else:
            frames = self.generate_full_source(landmarks, img, uid, eye, lmk)
        print({f"{time()-s:.3f}"})
        if preview:
            return [preview_clip]
        else:
            return [frames]

    def generate_preview(self, landmarks, img, uid, eye, lmk):
        frames = []
        for i, (p, y, m, e) in enumerate(landmarks):
            if i not in _preset_motion_unique:
                frames.append(None)
                continue
            _, frame = self.execute_image2(
                input_eye_ratio=eye,
                input_lip_ratio=m,
                input_head_pitch_variation=p,
                input_head_yaw_variation=y,
                input_head_roll_variation=0,
                input_image=img,
                retargeting_source_scale=1.0,
                flag_do_crop=True,
                lmk=lmk
            )
            cv2.imwrite(f"tmp/{uid}_{i}.webp", frame, [int(cv2.IMWRITE_WEBP_QUALITY), 20])
            frames.append(f"tmp/{uid}_{i}.webp")

        target_frames = [cv2.imread(f"tmp/{uid}_{i}.webp") for i in _preset_motion]
        images2video(target_frames, f"preview_mute_{uid}.mp4", fps=25)
        os.system(f"ffmpeg -y -i preview_mute_{uid}.mp4 -i char_example.wav -c:v copy -c:a libmp3lame -strict experimental -map 0:v:0 -map 1:a:0 preview_{uid}.mp4")
        with open(f"preview_{uid}.mp4", "rb") as fr:
            d = str(fr.read())
        return d

    def generate_full_source(self, landmarks, img, uid, eye, lmk):
        frames = []
        for i, (p, y, m, e) in enumerate(landmarks):
            _, frame = self.execute_image2(
                input_eye_ratio=eye,
                input_lip_ratio=m,
                input_head_pitch_variation=p,
                input_head_yaw_variation=y,
                input_head_roll_variation=0,
                input_image=img,
                retargeting_source_scale=1.0,
                flag_do_crop=True,
                lmk=lmk
            )
            with to_cv2(frame, f"tmp/{uid}_{i}.webp") as data:
                frames.append(str(data))
        return frames

character_handler = CharacterHandler.bind()

# ray start --head (unless ray is on)
#

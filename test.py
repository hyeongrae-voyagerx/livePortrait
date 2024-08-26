from src.live_portrait_pipeline import LivePortraitPipeline

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig

from src.utils.io import load_image_rgb

# # --- live portrait ---
args = ArgumentConfig()
inference_cfg = InferenceConfig()
crop_cfg = CropConfig()

lp = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
img = "assets/examples/source/hobbes.jpg"
img_rgb = load_image_rgb(img)
piui = lp.cropper.crop_source_image(img_rgb, lp.cropper.crop_cfg)
# lmk = piui["lmk_crop"]
# lp.live_portrait_wrapper.calc_combined_eye_ratio([[0.2]], lmk)
# lmk_256 = piui["lmk_crop_256x256"]

# --- face_alignment(MIT license) ---
import face_alignment
from skimage import io

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector=item)


# preds, pts = fa.get_landmarks(img_rgb)

lp.execute_timefn(img)

breakpoint()

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

from src.live_portrait_character import LivePortraitCharacter

def main():
    # initialize
    inference_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    model = LivePortraitCharacter(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    model.generate_face_images("assets/examples/source/hobbes.jpg")

if __name__ == "__main__":
    main()

# coding: utf-8

"""
Pipeline of LivePortrait
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp

from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video
from .utils.crop import prepare_paste_back, paste_back
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper

from .utils.io import load_img_online
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

from motion import Trajectory, AudioProcessor

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitCharacter:
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def generate_face_images(self, input_img_path, do_test=False):
        eye = self.get_initial_eye(input_img_path)
        trajectory = Trajectory(0, init_eye=eye)
        landmarks = trajectory.graph["landmarks"]
        frames = []
        from time import time
        s = time()
        for i, (p, y, m, e) in enumerate(landmarks):
            _, frame = self.execute_image2(
                input_eye_ratio=e,
                input_lip_ratio=m,
                input_head_pitch_variation=p,
                input_head_yaw_variation=y,
                input_head_roll_variation=0,
                input_image=input_img_path,
                retargeting_source_scale=1.0,
                flag_do_crop=True
            )
            frames.append(frame)
            cv2.imwrite(f"preset/test_webp/frame_{i}.webp", frame, [int(cv2.IMWRITE_WEBP_QUALITY), 20])
            print(f"\r{i} / {len(landmarks)}", end="")
        torch.save(trajectory, f"preset/test_webp/trajectory.pkl")
        print(f"| elapsed time for generating source images: {time()-s:.2f}")
        if do_test:
            print("start test")
            a = AudioProcessor()
            is_speaking = a("korean.wav")
            is_speaking = torch.nn.functional.pad(is_speaking, (0, 13), mode="constant", value=is_speaking[-1])[13:]
            traj = trajectory(is_speaking)
            target_frames2 = [cv2.imread("preset/test_webp/frame_{}.webp".format(f)) for f in traj]
            images2video(target_frames2, "test.mp4", fps=25)
            os.system(f"ffmpeg -y -i test.mp4 -i korean.wav -c:v copy -c:a libmp3lame -strict experimental -map 0:v:0 -map 1:a:0 test_audio.mp4")

        # legacy ai my character
        # preset = torch.load(f"preset/hobbes/frames.pkl")
        # t = preset["trajectory"]
        # frames = np.stack(preset["frames"])
        # for i, f in enumerate(frames):
        #     cv2.imwrite(f"preset/hobbes_webp/frame_{i}.webp", f, [int(cv2.IMWRITE_WEBP_QUALITY), 20])
        # trajectory = t(is_speaking)
        # target_frames = frames[trajectory]
        # images2video(target_frames, output_wo_audio, fps=25)
        # / legacy ai my character
        t2 = torch.load(f"preset/hobbes/traj.pkl")
        # target_frames2 = [cv2.imread("preset/hobbes_webp/frame_{}.webp".format(f)) for f in trajectory2]

        # pitch_seq = piui[:, 0].numpy()
        # yaw_seq = piui[:, 1].numpy()
        # mouth_seq = piui[:, 2].numpy()
        # # pitch_seq = (2.3*np.sin(time*2.7)+ 0.77*np.sin(time*8)).astype(float)
        # # yaw_seq = (1.9*np.sin(time*2) + 0.87*np.sin(time*6)).astype(float)
        # #roll_seq = (2*np.sin(time*1.9) + 0.67*np.sin(time*9)).astype(float)
        # roll_seq = np.zeros_like(yaw_seq)
        # # mouth_seq = (0.1*np.sin(time*11) + 0.031*np.sin(time*15)+0.01*np.sin(time*30)+0.171).astype(float)

        # log(f"Load source image from {input_img_path}.")
        # frames = []
        # for p, y, r, m in zip(pitch_seq, yaw_seq, roll_seq, mouth_seq):
        #     _, frame = self.execute_image2(
        #         input_eye_ratio=eye,
        #         input_lip_ratio=m,
        #         input_head_pitch_variation=p,
        #         input_head_yaw_variation=y,
        #         input_head_roll_variation=r,
        #         input_image=input_img_path,
        #         retargeting_source_scale=1.0,
        #         flag_do_crop=True
        #     )
        #     frames.append(frame)
        # extension = "." +input_img_path.split(".")[-1]
        # output = osp.join("animations", f"blahblah_{int(seconds)}_" + osp.basename(input_img_path).replace(extension, ".mp4"))
        # images2video(frames, "piui.mp4", fps=25)

    @torch.no_grad()
    def execute_image2(self, input_eye_ratio: float, input_lip_ratio: float, input_head_pitch_variation: float, input_head_yaw_variation: float, input_head_roll_variation: float, input_image, retargeting_source_scale: float, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_head_pitch_variation is None or input_head_yaw_variation is None or input_head_roll_variation is None:
            raise gr.Error("Invalid relative pose input 💥!", duration=5)
        # disposable feature
        f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
            self.prepare_retargeting2(input_image, input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation, retargeting_source_scale, flag_do_crop)
        if input_eye_ratio is None or input_lip_ratio is None:
            raise gr.Error("Invalid ratio input 💥!", duration=5)
        else:
            device = self.live_portrait_wrapper.device
            # inference_cfg = self.live_portrait_wrapper.inference_cfg
            x_s_user = x_s_user.to(device)
            f_s_user = f_s_user.to(device)
            R_s_user = R_s_user.to(device)
            R_d_user = R_d_user.to(device)

            x_c_s = x_s_info['kp'].to(device)
            delta_new = x_s_info['exp'].to(device)
            scale_new = x_s_info['scale'].to(device)
            t_new = x_s_info['t'].to(device)
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user
            delta_new = self.update_delta_new_lip_open(input_lip_ratio, delta_new)

            x_d_new = scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
            combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio([[float(input_eye_ratio)]], source_lmk_user)
            eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor)
            # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
            combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio([[float(0.01)]], source_lmk_user)
            lip_delta = self.live_portrait_wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor)
            x_d_new = x_d_new + eyes_delta + lip_delta
            x_d_new = self.live_portrait_wrapper.stitching(x_s_user, x_d_new)
            # D(W(f_s; x_s, x′_d))
            out = self.live_portrait_wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
            out = self.live_portrait_wrapper.parse_output(out['out'])[0]
            out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori)
            return out, out_to_ori_blend

    @torch.no_grad()
    def prepare_retargeting2(self, input_image, input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation, retargeting_source_scale, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            # gr.Info("Upload successfully!", duration=2)
            args_user = {'scale': retargeting_source_scale}
            # self.args = update_args(self.args, args_user)
            # self.cropper.update_config(self.args.__dict__)
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=2)
            crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
            if flag_do_crop:
                I_s = self.live_portrait_wrapper.prepare_source(crop_info['img_crop_256x256'])
            else:
                I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
            x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
            x_s_info_user_pitch = x_s_info['pitch'] + input_head_pitch_variation
            x_s_info_user_yaw = x_s_info['yaw'] + input_head_yaw_variation
            x_s_info_user_roll = x_s_info['roll'] + input_head_roll_variation
            R_s_user = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = get_rotation_matrix(x_s_info_user_pitch, x_s_info_user_yaw, x_s_info_user_roll)
            ############################################
            f_s_user = self.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s_user = self.live_portrait_wrapper.transform_keypoint(x_s_info)
            source_lmk_user = crop_info['lmk_crop']
            crop_M_c2o = crop_info['M_c2o']
            mask_ori = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            return f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            # when press the clear button, go here
            raise ValueError("Please upload a source portrait as the retargeting input 🤗🤗🤗")


    def init_retargeting2(self, retargeting_source_scale: float, input_image = None):
        """ initialize the retargeting slider
        """
        if input_image != None:
            inference_cfg = self.live_portrait_wrapper.inference_cfg
            ######## process source portrait ########
            img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=16)
            log(f"Load source image from {input_image}.")
            crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
            if crop_info is None:
                raise ValueError("Source portrait NO face detected")
            source_eye_ratio = calc_eye_close_ratio(crop_info['lmk_crop'][None])
            source_lip_ratio = calc_lip_close_ratio(crop_info['lmk_crop'][None])
            return round(float(source_eye_ratio.mean()), 2), round(source_lip_ratio[0][0], 2)
        return 0., 0.

    def get_initial_eye(self, input_image):
        img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=16)
        crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
        source_eye_ratio = calc_eye_close_ratio(crop_info['lmk_crop'][None])
        return round(float(source_eye_ratio.mean()), 2)

    @torch.no_grad()
    def update_delta_new_lip_open(self, lip_open, delta_new, **kwargs):
        delta_new[0, 19, 1] += lip_open * 0.001 * 120
        delta_new[0, 19, 2] += lip_open * 0.0001 * 120
        delta_new[0, 17, 1] += lip_open * -0.0001 * 120

        return delta_new

    def execute_timefn(self, filepath, audio_file):
        eye, _ = self.init_retargeting2(0.0, filepath)
        # time = np.arange(seconds*25) / 20
        a = AudioProcessor()
        # audio_path = "lpsample.wav"
        is_speaking = a(audio_file)
        # t = Trajectory(0)
        is_speaking = torch.nn.functional.pad(is_speaking, (0, 13), mode="constant", value=is_speaking[-1])[13:]
        # piui = t(is_speaking)
        # piui = t.graph["landmarks"][piui]
        # frames = []
        # for i, (p, y, m) in enumerate(t.graph["landmarks"]):
        #     _, frame = self.execute_image2(
        #         input_eye_ratio=eye,
        #         input_lip_ratio=m,
        #         input_head_pitch_variation=p,
        #         input_head_yaw_variation=y,
        #         input_head_roll_variation=0,
        #         input_image=filepath,
        #         retargeting_source_scale=1.0,
        #         flag_do_crop=True
        #     )
        #     frames.append(frame)
        #     print(f"\r{i}", end="")
        # torch.save({"frames": frames, "trajectory": t}, f"preset/hobbes/frames.pkl")

        # legacy ai my character
        # preset = torch.load(f"preset/hobbes/frames.pkl")
        # t = preset["trajectory"]
        # frames = np.stack(preset["frames"])
        # for i, f in enumerate(frames):
        #     cv2.imwrite(f"preset/hobbes_webp/frame_{i}.webp", f, [int(cv2.IMWRITE_WEBP_QUALITY), 20])
        # trajectory = t(is_speaking)
        # target_frames = frames[trajectory]
        output_wo_audio = osp.join("animations", f"{osp.basename(audio_file).split('.')[0]}_mute.mp4")
        # images2video(target_frames, output_wo_audio, fps=25)
        # / legacy ai my character
        t2 = torch.load(f"preset/hobbes/traj.pkl")
        trajectory2 = t2(is_speaking)
        target_frames2 = [cv2.imread("preset/hobbes_webp/frame_{}.webp".format(f)) for f in trajectory2]
        images2video(target_frames2, output_wo_audio, fps=25)

        # pitch_seq = piui[:, 0].numpy()
        # yaw_seq = piui[:, 1].numpy()
        # mouth_seq = piui[:, 2].numpy()
        # # pitch_seq = (2.3*np.sin(time*2.7)+ 0.77*np.sin(time*8)).astype(float)
        # # yaw_seq = (1.9*np.sin(time*2) + 0.87*np.sin(time*6)).astype(float)
        # #roll_seq = (2*np.sin(time*1.9) + 0.67*np.sin(time*9)).astype(float)
        # roll_seq = np.zeros_like(yaw_seq)
        # # mouth_seq = (0.1*np.sin(time*11) + 0.031*np.sin(time*15)+0.01*np.sin(time*30)+0.171).astype(float)

        # log(f"Load source image from {filepath}.")
        # frames = []
        # for p, y, r, m in zip(pitch_seq, yaw_seq, roll_seq, mouth_seq):
        #     _, frame = self.execute_image2(
        #         input_eye_ratio=eye,
        #         input_lip_ratio=m,
        #         input_head_pitch_variation=p,
        #         input_head_yaw_variation=y,
        #         input_head_roll_variation=r,
        #         input_image=filepath,
        #         retargeting_source_scale=1.0,
        #         flag_do_crop=True
        #     )
        #     frames.append(frame)
        # extension = "." +filepath.split(".")[-1]
        # output = osp.join("animations", f"blahblah_{int(seconds)}_" + osp.basename(filepath).replace(extension, ".mp4"))
        # images2video(frames, "piui.mp4", fps=25)
        output_w_audio = output_wo_audio.replace("mute", "audio")
        os.system(f"ffmpeg -y -i {output_wo_audio} -i {audio_file} -c:v copy -c:a libmp3lame -strict experimental -map 0:v:0 -map 1:a:0 {output_w_audio}")
        return output_w_audio

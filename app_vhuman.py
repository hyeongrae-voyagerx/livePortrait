# coding: utf-8

"""
The entrance of the gradio
"""

import os
import tyro
import subprocess
import gradio as gr
import os.path as osp
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig
# global_tab_selection = None

gradio_pipeline = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)

if args.gradio_temp_dir not in (None, ''):
    os.environ["GRADIO_TEMP_DIR"] = args.gradio_temp_dir
    os.makedirs(args.gradio_temp_dir, exist_ok=True)

def gpu_wrapped_execute_video(*args, **kwargs):
    if not isinstance(args[2], int) and False:
        args = args[:-1]
        return gradio_pipeline.execute_video(*args, **kwargs)
    else:
        output = gradio_pipeline.execute_timefn(args[0], audio_file=args[-1])
        return output, output


def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


# assets
title_md = "assets/gradio/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples_i2v = [
    [osp.join(example_portrait_dir, "s9.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s6.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s10.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s5.jpg"), osp.join(example_video_dir, "d18.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s7.jpg"), osp.join(example_video_dir, "d19.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s2.jpg"), osp.join(example_video_dir, "d13.mp4"), True, True, True, True],
]
data_examples_v2v = [
    [osp.join(example_portrait_dir, "s13.mp4"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False, False, 3e-7],
    # [osp.join(example_portrait_dir, "s14.mp4"), osp.join(example_video_dir, "d18.mp4"), True, True, True, False, False, 3e-7],
    # [osp.join(example_portrait_dir, "s15.mp4"), osp.join(example_video_dir, "d19.mp4"), True, True, True, False, False, 3e-7],
    [osp.join(example_portrait_dir, "s18.mp4"), osp.join(example_video_dir, "d6.mp4"), True, True, True, False, False, 3e-7],
    # [osp.join(example_portrait_dir, "s19.mp4"), osp.join(example_video_dir, "d6.mp4"), True, True, True, False, False, 3e-7],
    [osp.join(example_portrait_dir, "s20.mp4"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False, False, 3e-7],
]
#################### interface logic ####################

# Define components first
retargeting_source_scale = gr.Number(minimum=1.8, maximum=3.2, value=2.5, step=0.05, label="crop scale")
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
head_pitch_slider = gr.Slider(minimum=-15.0, maximum=15.0, value=0, step=1, label="relative pitch")
head_yaw_slider = gr.Slider(minimum=-25, maximum=25, value=0, step=1, label="relative yaw")
head_roll_slider = gr.Slider(minimum=-15.0, maximum=15.0, value=0, step=1, label="relative roll")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
output_video_i2v = gr.Video(autoplay=False)
output_video_concat_i2v = gr.Video(autoplay=False)


with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    gr.HTML(load_description(title_md))

    gr.Markdown(load_description("assets/gradio/gradio_description_upload.md"))
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Source Audio") as tab_audio:
                    source_audio = gr.Audio(type="filepath")
                    gr.Examples(
                        examples=["korean.wav", "english.wav", "chinese.wav"],
                        inputs=[source_audio],
                        cache_examples=False
                    )
                    with gr.Row():
                        process_button_animation = gr.Button("🚀 Animate", variant="primary")
                with gr.TabItem("🖼️ Source Image", visible=False) as tab_image:
                    with gr.Accordion(open=True, label="Source Image"):
                        source_image_input = gr.Image(type="filepath")
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s9.jpg")],
                                [osp.join(example_portrait_dir, "s7.jpg")],
                                [osp.join(example_portrait_dir, "s10.jpg")],
                                [osp.join(example_portrait_dir, "s22.jpg")],
                            ],
                            inputs=[source_image_input],
                            cache_examples=False,
                        )

                with gr.TabItem("🎞️ Source Video", visible=False) as tab_video:
                    with gr.Accordion(open=True, label="Source Video"):
                        source_video_input = gr.Video()
                        gr.Examples(
                            examples=[
                                [osp.join(example_portrait_dir, "s13.mp4")],
                                # [osp.join(example_portrait_dir, "s14.mp4")],
                                # [osp.join(example_portrait_dir, "s15.mp4")],
                                [osp.join(example_portrait_dir, "s18.mp4")],
                                # [osp.join(example_portrait_dir, "s19.mp4")],
                                [osp.join(example_portrait_dir, "s20.mp4")],
                            ],
                            inputs=[source_video_input],
                            cache_examples=False,
                        )

                tab_selection = gr.Textbox(visible=False)
                tab_image.select(lambda: "Image", None, tab_selection)
                tab_video.select(lambda: "Video", None, tab_selection)
            with gr.Accordion(open=True, label="Cropping Options for Source Image or Video", visible=False):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="do crop (source)")
                    scale = gr.Number(value=2.3, label="source crop scale", minimum=1.8, maximum=3.2, step=0.05)
                    vx_ratio = gr.Number(value=0.0, label="source crop x", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="source crop y", minimum=-0.5, maximum=0.5, step=0.01)
        with gr.Column():
            with gr.Row():
                with gr.Column(visible=False):
                    action_selection = gr.Radio(
                        choices=[
                            ("미소", osp.join(example_video_dir, "smile.webm")),
                            ("당황", osp.join(example_video_dir, "surprised.webm")),
                            ("끄덕끄덕", osp.join(example_video_dir, "gdgd.webm")),
                            ("어쩌구저쩌구", 10)],
                        label="Select Emotion"
                    )
                with gr.Column(visible=False):
                    blahblah_time = gr.Slider(minimum=2, maximum=21, value=5, label="어쩌구저쩌구_초(seconds)")

            with gr.Row():
                with gr.Accordion(open=True, label="The animated video in the original image space"):
                    output_video_i2v.render()


    with gr.Row():
        with gr.Accordion(open=True, label="Driving Video", visible=False):
            driving_video_input = gr.Video()
            gr.Examples(
                examples=[
                    [osp.join(example_video_dir, "d0.mp4")],
                    [osp.join(example_video_dir, "d18.mp4")],
                    [osp.join(example_video_dir, "d19.mp4")],
                    [osp.join(example_video_dir, "d14.mp4")],
                    [osp.join(example_video_dir, "d6.mp4")],
                    [osp.join(example_video_dir, "d20.mp4")],
                ],
                inputs=[driving_video_input],
                cache_examples=False,
            )
        with gr.Accordion(open=False, label="Animation Instructions", visible=False):
            gr.Markdown(load_description("assets/gradio/gradio_description_animation.md"))
        with gr.Accordion(open=True, label="Cropping Options for Driving Video", visible=False):
            with gr.Row():
                flag_crop_driving_video_input = gr.Checkbox(value=False, label="do crop (driving)")
                scale_crop_driving_video = gr.Number(value=2.2, label="driving crop scale", minimum=1.8, maximum=3.2, step=0.05)
                vx_ratio_crop_driving_video = gr.Number(value=0.0, label="driving crop x", minimum=-0.5, maximum=0.5, step=0.01)
                vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="driving crop y", minimum=-0.5, maximum=0.5, step=0.01)

    with gr.Row(visible=False):
        with gr.Accordion(open=True, label="Animation Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=True, label="relative motion")
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                flag_video_editing_head_rotation = gr.Checkbox(value=False, label="relative head rotation (v2v)")
                driving_smooth_observation_variance = gr.Number(value=3e-7, label="motion smooth strength (v2v)", minimum=1e-11, maximum=1e-2, step=1e-8)

    # gr.Markdown(load_description("assets/gradio/gradio_description_animate_clear.md"))
    with gr.Row():
        with gr.Column(visible=False):
            with gr.Accordion(open=True, label="The animated video"):
                output_video_concat_i2v.render()
    with gr.Row():
        process_button_reset = gr.ClearButton([source_image_input, source_video_input, action_selection, output_video_i2v, output_video_concat_i2v], value="🧹 Clear")

    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            source_image_input,
            source_video_input,
            action_selection,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            flag_crop_driving_video_input,
            flag_video_editing_head_rotation,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            driving_smooth_observation_variance,
            tab_selection,
            source_audio,
        ],
        outputs=[output_video_i2v, output_video_concat_i2v],
        show_progress=True
    )

    retargeting_input_image.change(
        fn=gradio_pipeline.init_retargeting,
        inputs=[retargeting_source_scale, retargeting_input_image],
        outputs=[eye_retargeting_slider, lip_retargeting_slider]
    )

demo.launch(
    server_port=args.server_port,
    share=args.share,
    server_name="0.0.0.0",
)

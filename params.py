import ast
from typing import Dict, Tuple, Union
import numpy as np


# from imgtag import ImgTag  # metadata
# from libxmp import *  # metadata
# import libxmp  # metadata

# import json
import pandas as pd
import yaml

from conf._types import Config

import argparse
import requests
import os
from tqdm import tqdm
import re

model_dir = "models"


def read_param(yaml_path) -> Tuple[Config, argparse.Namespace, Dict[str, pd.Series]]:
    # if model_dir does not exist, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    # convert params to Config object
    config = Config(**params)

    if config.initial_image != "":
        print(
            "WARNING: You have specified an initial image. Note that the image resolution "
            "will be inherited from this image, not whatever width and height you specified. "
            "If the initial image resolution is too high, this can result in out of memory errors."
        )
    elif config.width * config.height > 160000:
        print(
            "WARNING: The width and height you have specified may be too high, in which case "
            "you will encounter out of memory errors either at the image generation stage or the "
            "video synthesis stage. If so, try reducing the resolution"
        )

    model_names = {
        "vqgan_imagenet_f16_16384": "ImageNet 16384",
        "vqgan_imagenet_f16_1024": "ImageNet 1024",
        "wikiart_1024": "WikiArt 1024",
        "wikiart_16384": "WikiArt 16384",
        "coco": "COCO-Stuff",
        "faceshq": "FacesHQ",
        "sflckr": "S-FLCKR",
    }
    model_name = model_names[config.model]

    # Download the model files
    yaml_url = f"https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
    ckpt_url = f"https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
    if not os.path.exists(f"{model_dir}/{config.model}.yaml"):
        yaml_response = requests.get(yaml_url, stream=True)
        if yaml_response.status_code == 200:
            pbar = tqdm(
                total=int(yaml_response.headers["Content-Length"]),
                unit="B",
                unit_scale=True,
            )
            with open(f"{model_dir}/{config.model}.yaml", "wb") as f:
                for chunk in yaml_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()

    if not os.path.exists(f"{model_dir}/{config.model}.ckpt"):
        ckpt_response = requests.get(ckpt_url, stream=True)
        if ckpt_response.status_code == 200:
            pbar = tqdm(
                total=int(ckpt_response.headers["Content-Length"]),
                unit="B",
                unit_scale=True,
            )
            with open(f"{model_dir}/{config.model}.ckpt", "wb") as f:
                for chunk in ckpt_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()

    if config.seed == -1:
        config.seed = None

    def parse_key_frames(string, prompt_parser=None) -> Dict[int, str]:
        """Given a string representing frame numbers paired with parameter values at that frame,
        return a dictionary with the frame numbers as keys and the parameter values as the values.

        Parameters
        ----------
        string: string
            Frame numbers paired with parameter values at that frame number, in the format
            'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
        prompt_parser: function or None, optional
            If provided, prompt_parser will be applied to each string of parameter values.

        Returns
        -------
        dict
            Frame numbers as keys, parameter values at that frame number as values

        Raises
        ------
        RuntimeError
            If the input string does not match the expected format.

        Examples
        --------
        >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
        {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

        >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
        {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
        """

        try:
            # This is the preferred way, the regex way will eventually be deprecated.
            frames = ast.literal_eval("{" + string + "}")
            if isinstance(frames, set):
                # If user forgot keyframes, just set value of frame 0
                (frame,) = list(frames)
                frames = {0: frame}
            return frames
        except Exception:
            import re

            pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
            frames = dict()
            for match_object in re.finditer(pattern, string):
                frame = int(match_object.groupdict()["frame"])
                param = match_object.groupdict()["param"]
                if prompt_parser:
                    frames[frame] = prompt_parser(param)
                else:
                    frames[frame] = param

            if frames == {} and len(string) != 0:
                raise RuntimeError(
                    f"Key Frame string not correctly formatted: {string}"
                )
            return frames

    # Defaults, if left empty
    # if config.angle == "":
    #     angle = "0"
    # if config.zoom == "":
    #     zoom = "1"
    # if config.translation_x == "":
    #     translation_x = "0"
    # if config.translation_y == "":
    #     translation_y = "0"
    # if config.iterations_per_frame == "":
    #     iterations_per_frame = "10"

    parameter_dicts = dict()
    # parameter_dicts["zoom"] = parse_key_frames(zoom, prompt_parser=float)
    # parameter_dicts["angle"] = parse_key_frames(angle, prompt_parser=float)
    # parameter_dicts["translation_x"] = parse_key_frames(
    #     translation_x, prompt_parser=float
    # )
    # parameter_dicts["translation_y"] = parse_key_frames(
    #     translation_y, prompt_parser=float
    # )
    # parameter_dicts["iterations_per_frame"] = parse_key_frames(
    #     iterations_per_frame, prompt_parser=int
    # )

    # check config.zoom is Dict[str, float] or Dict[str, Tuple[float, float]]
    if isinstance(config.zoom, dict) and len(config.zoom.values()) > 0:
        ele = list(config.zoom.values())[0]
        if isinstance(ele, float) or isinstance(ele, int):
            parameter_dicts["zoom_x"] = config.zoom
            parameter_dicts["zoom_y"] = config.zoom
        elif isinstance(ele, list):
            parameter_dicts["zoom_x"] = {k: v[0] for k, v in config.zoom.items()}  # type: ignore
            parameter_dicts["zoom_y"] = {k: v[1] for k, v in config.zoom.items()}  # type: ignore
    else:
        parameter_dicts["zoom_x"] = 1
        parameter_dicts["zoom_y"] = 1
    parameter_dicts["angle"] = config.angle
    parameter_dicts["translation_x"] = config.translation_x
    parameter_dicts["translation_y"] = config.translation_y
    if isinstance(config.iterations_per_frame, int):
        parameter_dicts["iterations_per_frame"] = {0: config.iterations_per_frame}
    else:
        parameter_dicts["iterations_per_frame"] = config.iterations_per_frame

    # text_prompts_dict = parse_key_frames(config.text_prompts)
    text_prompts_dict: Dict[str, Dict[str, float]] = {
        data["prompt"]: data["keyframes"] for data in config.text_prompts
    }
    if all([isinstance(value, dict) for value in list(text_prompts_dict.values())]):
        for key, value in list(text_prompts_dict.items()):
            parameter_dicts[f"text_prompt: {key}"] = value
    else:
        raise ValueError(
            f"Text prompts must be in keyframes format: {text_prompts_dict}"
        )
        # Old format
        # text_prompts_dict = parse_key_frames(
        #     config.text_prompts, prompt_parser=lambda x: x.split("|")
        # )
        # for frame, prompt_list in text_prompts_dict.items():
        #     for prompt in prompt_list:
        #         prompt_key, prompt_value = prompt.split(":")
        #         prompt_key = f"text_prompt: {prompt_key.strip()}"
        #         prompt_value = prompt_value.strip()
        #         if prompt_key not in parameter_dicts:
        #             parameter_dicts[prompt_key] = dict()
        #         parameter_dicts[prompt_key][frame] = prompt_value

    # image_prompts_dict = parse_key_frames(config.target_images)
    image_prompts_dict = {
        data["image_path"]: data["keyframes"] for data in config.target_images
    }
    if all([isinstance(value, dict) for value in list(image_prompts_dict.values())]):
        for key, value in list(image_prompts_dict.items()):
            parameter_dicts[f"image_prompt: {key}"] = value
    else:
        raise ValueError(
            f"Image prompts must be in keyframes format: {image_prompts_dict}"
        )
        # Old format
        # image_prompts_dict = parse_key_frames(
        #     config.target_images, prompt_parser=lambda x: x.split("|")
        # )
        # for frame, prompt_list in image_prompts_dict.items():
        #     for prompt in prompt_list:
        #         prompt_key, prompt_value = prompt.split(":")
        #         prompt_key = f"image_prompt: {prompt_key.strip()}"
        #         prompt_value = prompt_value.strip()
        #         if prompt_key not in parameter_dicts:
        #             parameter_dicts[prompt_key] = dict()
        #         parameter_dicts[prompt_key][frame] = prompt_value

    args = argparse.Namespace(
        prompts=config.text_prompts,
        image_prompts=config.target_images,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[config.width, config.height],
        init_weight=0.0,
        clip_model="ViT-B/32",
        vqgan_config=f"{model_dir}/{config.model}.yaml",
        vqgan_checkpoint=f"{model_dir}/{config.model}.ckpt",
        step_size=0.1,
        cutn=64,
        cut_pow=1.0,
        display_freq=config.interval,
        seed=config.seed,
    )

    def get_inbetweens(key_frames_dict, integer=False):
        """Given a dict with frame numbers as keys and a parameter value as values,
        return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
        Any values not provided in the input dict are calculated by linear interpolation between
        the values of the previous and next provided frames. If there is no previous provided frame, then
        the value is equal to the value of the next provided frame, or if there is no next provided frame,
        then the value is equal to the value of the previous provided frame. If no frames are provided,
        all frame values are NaN.

        Parameters
        ----------
        key_frames_dict: dict
            A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
        integer: Bool, optional
            If True, the values of the output series are converted to integers.
            Otherwise, the values are floats.

        Returns
        -------
        pd.Series
            A Series with length max_frames representing the parameter values for each frame.

        Examples
        --------
        >>> max_frames = 5
        >>> get_inbetweens({1: 5, 3: 6})
        0    5.0
        1    5.0
        2    5.5
        3    6.0
        4    6.0
        dtype: float64

        >>> get_inbetweens({1: 5, 3: 6}, integer=True)
        0    5
        1    5
        2    5
        3    6
        4    6
        dtype: int64
        """
        key_frame_series = pd.Series([np.nan for a in range(config.max_frames)])
        for i, value in key_frames_dict.items():
            key_frame_series[i] = value
        key_frame_series = key_frame_series.astype(float)
        key_frame_series = key_frame_series.interpolate(limit_direction="both")
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    text_prompts_series_dict = dict()
    for parameter in parameter_dicts.keys():
        if len(parameter_dicts[parameter]) > 0:
            if parameter.startswith("text_prompt:"):
                text_prompts_series_dict[parameter] = get_inbetweens(
                    parameter_dicts[parameter]
                )
    text_prompts_series = pd.Series([np.nan for a in range(config.max_frames)])
    for i in range(config.max_frames):
        combined_prompt = []
        for parameter, value in text_prompts_series_dict.items():
            parameter = parameter[len("text_prompt:") :].strip()
            combined_prompt.append(f"{parameter}: {value[i]}")
        text_prompts_series[i] = " | ".join(combined_prompt)

    image_prompts_series_dict = dict()
    for parameter in parameter_dicts.keys():
        if len(parameter_dicts[parameter]) > 0:
            if parameter.startswith("image_prompt:"):
                image_prompts_series_dict[parameter] = get_inbetweens(
                    parameter_dicts[parameter]
                )
    target_images_series = pd.Series([np.nan for a in range(config.max_frames)])
    for i in range(config.max_frames):
        combined_prompt = []
        for parameter, value in image_prompts_series_dict.items():
            parameter = parameter[len("image_prompt:") :].strip()
            combined_prompt.append(f"{parameter}: {value[i]}")
        target_images_series[i] = " | ".join(combined_prompt)

    angle_series = get_inbetweens(parameter_dicts["angle"])
    # zoom_series = get_inbetweens(parameter_dicts["zoom"])
    zoom_x_series = get_inbetweens(parameter_dicts["zoom_x"])
    zoom_y_series = get_inbetweens(parameter_dicts["zoom_y"])
    # for i, zoom in enumerate(zoom_series):
    #     if zoom <= 0:
    #         print(
    #             f"WARNING: You have selected a zoom of {zoom} at frame {i}. "
    #             "This is meaningless. "
    #             "If you want to zoom out, use a value between 0 and 1. "
    #             "If you want no zoom, use a value of 1."
    #         )
    for i, zoom in enumerate(zoom_x_series):
        if zoom <= 0:
            print(
                f"WARNING: You have selected a zoom of {zoom} at frame {i}. "
                "This is meaningless. "
                "If you want to zoom out, use a value between 0 and 1. "
                "If you want no zoom, use a value of 1."
            )
    for i, zoom in enumerate(zoom_y_series):
        if zoom <= 0:
            print(
                f"WARNING: You have selected a zoom of {zoom} at frame {i}. "
                "This is meaningless. "
                "If you want to zoom out, use a value between 0 and 1. "
                "If you want no zoom, use a value of 1."
            )

    translation_x_series = get_inbetweens(parameter_dicts["translation_x"])
    translation_y_series = get_inbetweens(parameter_dicts["translation_y"])

    iterations_per_frame_series = get_inbetweens(
        parameter_dicts["iterations_per_frame"], integer=True
    )

    return (
        config,
        args,
        {
            "text_prompts_series": text_prompts_series,
            "target_images_series": target_images_series,
            "angle_series": angle_series,
            # "zoom_series": zoom_series,
            "zoom_x_series": zoom_x_series,
            "zoom_y_series": zoom_y_series,
            "translation_x_series": translation_x_series,
            "translation_y_series": translation_y_series,
            "iterations_per_frame_series": iterations_per_frame_series,
        },
    )


if __name__ == "__main__":
    config, args, series_dict = read_param("conf/test.yaml")
    print(config)
    print(args)
    print(series_dict)

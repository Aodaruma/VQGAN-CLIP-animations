import argparse
from pathlib import Path
import sys
import os
import cv2
import pandas as pd
import numpy as np
import glob

working_dir = "./content"

sys.path.append("./taming-transformers")


from IPython import display
from PIL import Image
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

from CLIP import clip
import numpy as np
import imageio
from PIL import ImageFile, Image

# from imgtag import ImgTag  # metadata
# from libxmp import *  # metadata
# import libxmp  # metadata
# from stegano import lsb
# import json

from conf._types import Config

from funcs import *
from params import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args, config: Config, series):
    # for var in ["device", "model", "perceptor", "z"]:
    #     try:
    #         del globals()[var]
    #     except:
    #         pass

    try:
        import gc

        gc.collect()
    except:
        pass

    try:
        torch.cuda.empty_cache()
    except:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print("Using seed:", seed)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = (
        clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    )

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2 ** (model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
    stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete

    def read_image_workaround(path):
        """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
        this incompatibility to avoid colour inversions."""
        im_tmp = cv2.imread(path)
        return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

    for i in range(config.max_frames):
        if stop_on_next_loop:
            break

        text_prompts = series["text_prompts_series"][i]
        if isinstance(text_prompts, str):
            text_prompts = [phrase.strip() for phrase in text_prompts.split("|")]
            if text_prompts == [""]:
                text_prompts = []
        else:
            raise ValueError(f"text_prompts must be a string: {text_prompts}")
        args.prompts = config.text_prompts

        target_images = series["target_images_series"][i]

        if target_images == "None" or not target_images:
            target_images = []
        args.image_prompts = config.target_images

        angle = series["angle_series"][i]
        zoom = series["zoom_series"][i]
        translation_x = series["translation_x_series"][i]
        translation_y = series["translation_y_series"][i]
        iterations_per_frame = series["iterations_per_frame_series"][i]
        print(
            f"config.text_prompts: {config.text_prompts}",
            f"image_prompts: {config.target_images}",
            f"angle: {angle}",
            f"zoom: {zoom}",
            f"translation_x: {translation_x}",
            f"translation_y: {translation_y}",
            f"iterations_per_frame: {iterations_per_frame}",
        )
        try:
            if i == 0 and config.initial_image and config.initial_image != "":
                img_0 = read_image_workaround(config.initial_image)
                z, *_ = model.encode(
                    TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1
                )
            elif i == 0 and not os.path.isfile(f"{working_dir}/steps/{i:04d}.png"):
                one_hot = F.one_hot(
                    torch.randint(n_toks, [toksY * toksX], device=device), n_toks
                ).float()
                z = one_hot @ model.quantize.embedding.weight
                z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
            else:
                if config.save_all_iterations:
                    img_0 = read_image_workaround(
                        f"{working_dir}/steps/{i-1:04d}_{iterations_per_frame}.png"
                    )
                else:
                    img_0 = read_image_workaround(f"{working_dir}/steps/{i-1:04d}.png")

                center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
                trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])  # type: ignore
                rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)  # type: ignore

                trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                transformation_matrix = np.matmul(rot_mat, trans_mat)

                img_0 = cv2.warpPerspective(
                    img_0,
                    transformation_matrix,
                    (img_0.shape[1], img_0.shape[0]),
                    borderMode=cv2.BORDER_WRAP,
                )
                z, *_ = model.encode(
                    TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1
                )
            # i += 1

            z_orig = z.clone()
            z.requires_grad_(True)
            opt = optim.Adam([z], lr=args.step_size)

            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            pMs = []

            for prompt in series["text_prompts_series"][i].split("|"):
                prompt = prompt.strip()
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

            for prompt in series["target_images_series"][i].split("|"):
                prompt = prompt.strip()
                if prompt == "":
                    continue
                path, weight, stop = parse_prompt(prompt)
                img = resize_image(Image.open(path).convert("RGB"), (sideX, sideY))
                batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
                embed = perceptor.encode_image(normalize(batch)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

            for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
                gen = torch.Generator().manual_seed(seed)
                embed = torch.empty([1, perceptor.visual.output_dim]).normal_(
                    generator=gen
                )
                pMs.append(Prompt(embed, weight).to(device))

            def synth(z):
                z_q = vector_quantize(
                    z.movedim(1, 3), model.quantize.embedding.weight
                ).movedim(  # type: ignore
                    3, 1
                )  # type: ignore
                return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

            # def add_xmp_data(filename):
            #     imagen = ImgTag(filename=filename)
            #     imagen.xmp.append_array_item(
            #         libxmp.consts.XMP_NS_DC,
            #         "creator",
            #         "VQGAN+CLIP",
            #         {"prop_array_is_ordered": True, "prop_value_is_array": True},
            #     )
            #     if args.prompts:
            #         imagen.xmp.append_array_item(
            #             libxmp.consts.XMP_NS_DC,
            #             "title",
            #             " | ".join(args.prompts),
            #             {"prop_array_is_ordered": True, "prop_value_is_array": True},
            #         )
            #     else:
            #         imagen.xmp.append_array_item(
            #             libxmp.consts.XMP_NS_DC,
            #             "title",
            #             "None",
            #             {"prop_array_is_ordered": True, "prop_value_is_array": True},
            #         )
            #     imagen.xmp.append_array_item(
            #         libxmp.consts.XMP_NS_DC,
            #         "i",
            #         str(i),
            #         {"prop_array_is_ordered": True, "prop_value_is_array": True},
            #     )
            #     imagen.xmp.append_array_item(
            #         libxmp.consts.XMP_NS_DC,
            #         "model",
            #         model_name,
            #         {"prop_array_is_ordered": True, "prop_value_is_array": True},
            #     )
            #     imagen.xmp.append_array_item(
            #         libxmp.consts.XMP_NS_DC,
            #         "seed",
            #         str(seed),
            #         {"prop_array_is_ordered": True, "prop_value_is_array": True},
            #     )
            #     imagen.close()

            # def add_stegano_data(filename):
            #     data = {
            #         "title": " | ".join(args.prompts) if args.prompts else None,
            #         "notebook": "VQGAN+CLIP",
            #         "i": i,
            #         "model": model_name,
            #         "seed": str(seed),
            #     }
            #     lsb.hide(filename, json.dumps(data)).save(filename)

            @torch.no_grad()
            def checkin(i, losses):
                losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
                tqdm.write(
                    f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}"  # type: ignore
                )
                out = synth(z)
                TF.to_pil_image(out[0].cpu()).save("progress.png")  # type: ignore
                # add_stegano_data("progress.png")
                # add_xmp_data("progress.png")
                # display.display(display.Image("progress.png"))

            def save_output(i, img, suffix=None):
                filename = (
                    f"{working_dir}/steps/{i:04}{'_' + suffix if suffix else ''}.png"
                )
                imageio.imwrite(filename, np.array(img))
                # add_stegano_data(filename)
                # add_xmp_data(filename)

            def ascend_txt(i, save=True, suffix=None):
                out = synth(z)
                iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

                result = []

                if args.init_weight:
                    result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

                for prompt in pMs:
                    result.append(prompt(iii))
                img = np.array(
                    out.mul(255)  # type: ignore
                    .clamp(0, 255)[0]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.uint8)
                )[:, :, :]
                img = np.transpose(img, (1, 2, 0))
                if save:
                    save_output(i, img, suffix=suffix)
                return result

            def train(i, save=True, suffix=None):
                opt.zero_grad()
                lossAll = ascend_txt(i, save=save, suffix=suffix)
                if i % args.display_freq == 0 and save:
                    checkin(i, lossAll)
                loss = sum(lossAll)
                loss.backward()  # type: ignore
                opt.step()
                with torch.no_grad():
                    z.copy_(z.maximum(z_min).minimum(z_max))

            with tqdm(total=iterations_per_frame) as pbar:
                if iterations_per_frame == 0:
                    save_output(i, img_0)
                j = 1
                while True:
                    suffix = str(j) if config.save_all_iterations else None
                    if j >= iterations_per_frame:  # type: ignore
                        train(i, save=True, suffix=suffix)
                        break
                    if config.save_all_iterations:
                        train(i, save=True, suffix=suffix)
                    else:
                        train(i, save=False, suffix=suffix)
                    j += 1
                    pbar.update()
        except KeyboardInterrupt:
            stop_on_next_loop = True
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def check_if_yaml_exists(path: str):
        if not Path(path).is_file():
            raise argparse.ArgumentTypeError(f"{path} is not a valid file path")
        return path

    parser.add_argument(
        "config_path",
        type=check_if_yaml_exists,
        help="Path to the config of the VQGAN+CLIP animations (yaml)",
    )

    arguments = parser.parse_args()
    config_path = arguments.config_path

    config, args, series = read_param(config_path)

    # check is initial image path is provided as sequence images (e.g. image_*.png)
    if config.initial_image and "*" in config.initial_image:
        initial_images = sorted(glob.glob(config.initial_image))
        if initial_images:
            for i, img_path in enumerate(initial_images):
                print(
                    f"\n===== Processing initial image {img_path} ({i+1}/{len(initial_images)})"
                )
                if not os.path.isfile(img_path):
                    print(f"Initial image {img_path} not found, skipping")
                    continue
                config.initial_image = img_path
                main(args, config, series)
        else:
            raise ValueError(f"No images found in {Path(config.initial_image).parent}")
    else:
        main(args, config, series)

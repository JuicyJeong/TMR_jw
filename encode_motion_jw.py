import os
from omegaconf import DictConfig
import logging
import hydra

logger = logging.getLogger(__name__)


# @hydra.main(version_base=None, config_path="configs", config_name="encode_motion")
def encode_motion_jw(motion):
    # 외부에서 설정을 받아서 사용
    # if run_dir is not None:
    #     cfg.run_dir = run_dir
    # if npy_path is not None:
    #     cfg.npy = npy_path

    device = 'cuda'
    run_dir = 'models/tmr_humanml3d_guoh3dfeats'
    ckpt_name = 'last'

    # npy_path = cfg.npy

    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    ######################################################
    import warnings
    warnings.filterwarnings("ignore", message="No audio backend is available.")

    from pytorch_lightning import seed_everything

    ######################################################

    from src.data.collate import collate_x_dict

    cfg = read_config(run_dir)

    # logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)
    # # normalizer의 mean과 std를 GPU로 이동
    # normalizer.mean = normalizer.mean.to(device)
    # normalizer.std = normalizer.std.to(device)
    # motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    motion = normalizer(motion)
    motion = motion.to(device)

    motion_x_dict = {"x": motion, "length": len(motion)}

    seed_everything(cfg.seed)

    with torch.inference_mode():
        motion_x_dict = collate_x_dict([motion_x_dict])
        latent = model.encode(motion_x_dict, sample_mean=True)[0]
        latent = latent.cpu().numpy()

    return latent

    # fname = os.path.split(npy_path)[1]
    # output_folder = os.path.join(run_dir, "encoded")
    # os.makedirs(output_folder, exist_ok=True)
    # path = os.path.join(output_folder, fname)

    # np.save(path, latent)
    # logger.info(f"Encoding done, latent saved in:\n{path}")


# if __name__ == "__main__":
#     encode_motion()

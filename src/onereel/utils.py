"""Small helpers shared by the model adapters."""
from __future__ import annotations


def local_snapshot(repo: str) -> str:
    """path of a model already in the Hugging Face cache (no network)"""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def save_video(frames, path, fps: int = 24):
    """H.264 / yuv420p mp4 from float frames in [0, 1] (plays in browsers and VS Code)"""
    import imageio.v3 as iio
    import numpy as np
    arr = (np.clip(np.asarray(frames), 0, 1) * 255).round().astype(np.uint8)
    iio.imwrite(str(path), arr, fps=fps, codec='libx264', plugin='pyav', out_pixel_format='yuv420p')

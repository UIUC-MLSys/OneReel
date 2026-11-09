"""
eval/utils.py
Shared evaluation utilities:
  - get_video_paths(folder)                    : list video files in a folder
  - load_viclip(device)                        : load ViCLIP model
  - compute_viclip_similarity(path_a, path_b)  : ViCLIP cosine similarity between two videos
  - compute_vbench_scores(videos_path, ...)    : run VBench via subprocess, return scores dict
  - load_clip_model(device)                    : load CLIP ViT-L/14 model + processor
  - compute_clip_score(video_path, prompt)     : mean-frame CLIP cosine similarity (×100)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTS: set = {".mp4", ".avi", ".mov", ".gif"}
_SKIP_STEMS: set = {"combined"}

_VICLIP_SNAPSHOT = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / (
    "hub/models--OpenGVLab--ViCLIP/snapshots/67fbc34595ae6e5f56ab630fe421d3fc45c8670d"
)
_VICLIP_WEIGHTS = _VICLIP_SNAPSHOT / "ViClip-InternVid-10M-FLT.pth"

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_normalize = Normalize(_CLIP_MEAN, _CLIP_STD)

_VICLIP_T = 32        # frames to extract from a video for temporal coverage
_VICLIP_MODEL_T = 8   # ViCLIP temporal positional embedding size (fixed by model weights)


# ---------------------------------------------------------------------------
# Video file discovery
# ---------------------------------------------------------------------------

def get_video_paths(folder: str) -> List[Path]:
    """Return sorted video file paths from *folder* (non-recursive, skips 'combined')."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and p.stem.lower() not in _SKIP_STEMS
    )
    return paths


# ---------------------------------------------------------------------------
# ViCLIP model loading
# ---------------------------------------------------------------------------

def load_viclip(device: Optional[str] = None):
    """Load ViCLIP (InternVid-10M-FLT).  Auto-selects CUDA if *device* is None."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shim for removed pkg_resources (setuptools ≥82)
    if "pkg_resources" not in sys.modules:
        import types
        import packaging as _pkg
        _mock = types.ModuleType("pkg_resources")
        _mock.packaging = _pkg
        sys.modules["pkg_resources"] = _mock

    if str(_VICLIP_SNAPSHOT) not in sys.path:
        sys.path.insert(0, str(_VICLIP_SNAPSHOT))
    from viclip import ViCLIP  # noqa: PLC0415

    model = ViCLIP(pretrain=str(_VICLIP_WEIGHTS))
    model.eval().to(device)
    print(f"[ViCLIP] Loaded from {_VICLIP_WEIGHTS} on {device}")
    return model


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _bgr_to_pil(frame) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_frames_from_video(
    video_path: str,
    n: int = _VICLIP_T,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
) -> List[Image.Image]:
    """
    Extract *n* uniformly-sampled frames from [start_frac, end_frac] of *video_path*.
    Returns a list of PIL Images.
    """
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback: read all frames if total unknown
    if total <= 0:
        all_frames: List[Image.Image] = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            all_frames.append(_bgr_to_pil(f))
        cap.release()
        total = len(all_frames)
        if total == 0:
            raise ValueError(f"No frames in: {path}")
        i_start = int(total * start_frac)
        i_end   = max(i_start + 1, int(total * end_frac))
        window  = all_frames[i_start:i_end]
    else:
        i_start = int(total * start_frac)
        i_end   = max(i_start + 1, int(total * end_frac))
        window_len = i_end - i_start
        n_actual = min(n, window_len)
        idxs = [
            i_start + int(round(k * (window_len - 1) / max(n_actual - 1, 1)))
            for k in range(n_actual)
        ]
        window = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                window.append(_bgr_to_pil(f))
            else:
                warnings.warn(f"Could not read frame {idx} from {path}")
        cap.release()
        if not window:
            raise ValueError(f"No frames extracted from {path}")
        all_frames = None

    # Sample/pad to exactly n frames
    if all_frames is not None:
        # Already handled above for the fallback path
        pass

    frames = window
    nf = len(frames)
    if nf >= n:
        idxs = [int(round(k * (nf - 1) / max(n - 1, 1))) for k in range(n)]
        frames = [frames[i] for i in idxs]
    else:
        frames = frames + [frames[-1]] * (n - nf)

    return frames


# ---------------------------------------------------------------------------
# ViCLIP embedding
# ---------------------------------------------------------------------------

def _frames_to_tensor(frames: List[Image.Image]) -> torch.Tensor:
    """Convert list of PIL Images → (T, C, H, W) CLIP-normalised float tensor."""
    tensors = []
    for img in frames:
        t = (
            torch.from_numpy(np.array(img.convert("RGB")))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        t = Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=False)(t)
        t = CenterCrop(224)(t)
        t = _normalize(t)
        tensors.append(t)
    return torch.stack(tensors)  # (T, C, H, W)


def get_viclip_embedding(
    frames: List[Image.Image],
    model,
    device: str,
) -> torch.Tensor:
    """Return L2-normalised (1, D) ViCLIP embedding for a list of PIL frames.

    Regardless of how many frames are supplied, subsamples to exactly
    _VICLIP_MODEL_T (=8) frames before calling encode_vision, because the
    model's temporal positional embedding has a fixed size of 8.
    """
    n = len(frames)
    T = _VICLIP_MODEL_T   # model constraint: exactly 8 frames
    if n >= T:
        idxs = [int(round(k * (n - 1) / (T - 1))) for k in range(T)]
        frames = [frames[i] for i in idxs]
    else:
        frames = frames + [frames[-1]] * (T - n)

    clip_tensor = _frames_to_tensor(frames).unsqueeze(0).to(device)  # (1, 8, C, H, W)
    with torch.no_grad():
        feat = model.encode_vision(clip_tensor, test=True).float()
        feat = F.normalize(feat, dim=-1)
    return feat.cpu()


# ---------------------------------------------------------------------------
# Public similarity API
# ---------------------------------------------------------------------------

def compute_viclip_similarity(
    video_a_path: str,
    video_b_path: str,
    model=None,
    device: Optional[str] = None,
    n_frames: int = _VICLIP_T,
    a_start: float = 0.0,
    a_end: float = 1.0,
    b_start: float = 0.0,
    b_end: float = 1.0,
) -> float:
    """
    Compute ViCLIP cosine similarity between two video segments.

    *a_start/a_end* and *b_start/b_end* control which temporal fraction of each
    video is used (default: full clip).  Loads ViCLIP if *model* is None.
    Returns a float in [-1, 1].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_viclip(device)

    frames_a = extract_frames_from_video(video_a_path, n=n_frames, start_frac=a_start, end_frac=a_end)
    frames_b = extract_frames_from_video(video_b_path, n=n_frames, start_frac=b_start, end_frac=b_end)

    feat_a = get_viclip_embedding(frames_a, model, device)
    feat_b = get_viclip_embedding(frames_b, model, device)
    return float((feat_a @ feat_b.T).item())


def compute_intra_clip_consistency(
    video_path: str,
    model=None,
    device: Optional[str] = None,
    n_frames: int = _VICLIP_T,
) -> float:
    """
    Measure temporal consistency within a single clip by computing ViCLIP similarity
    between its first half and second half.  Higher = more consistent.
    """
    return compute_viclip_similarity(
        video_path, video_path,
        model=model, device=device, n_frames=n_frames,
        a_start=0.0, a_end=0.5,
        b_start=0.5, b_end=1.0,
    )


# ---------------------------------------------------------------------------
# CSV helpers (used by run_vbench.py)
# ---------------------------------------------------------------------------

def load_csv(path: str) -> List[Dict]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: List[Dict], path: str) -> None:
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# VBench evaluation (subprocess)
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent


def compute_vbench_scores(
    videos_path: str,
    output_path: str,
    dimensions: Optional[List[str]] = None,
    mode: str = "custom_input",
) -> Dict[str, float]:
    """
    Run VBench evaluation on *videos_path* and return {dimension: score}.

    Launches eval/eval_vbench.py as a subprocess to avoid distributed-init
    conflicts with an already-running torch.distributed process group.
    *output_path* is the directory where VBench writes its JSON results.
    """
    if dimensions is None:
        dimensions = ["imaging_quality", "aesthetic_quality", "motion_smoothness", "subject_consistency", "background_consistency"]

    eval_script = _EVAL_DIR / "eval_vbench.py"
    if not eval_script.exists():
        print(f"[VBench] eval script not found: {eval_script}")
        return {}

    os.makedirs(output_path, exist_ok=True)

    cmd = [
        sys.executable, str(eval_script),
        "--videos_path", str(videos_path),
        "--output_path", str(output_path),
        "--mode", mode,
        "--dimension", *dimensions,
    ]
    print(f"[VBench] Running: {' '.join(cmd)}")

    # Propagate PYTHONPATH so the subprocess can import onereel etc.
    env = os.environ.copy()
    src_dir = str(_EVAL_DIR.parent / "src")
    env["PYTHONPATH"] = src_dir + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[VBench] STDERR:\n{result.stderr[-3000:]}")
        print(f"[VBench] STDOUT:\n{result.stdout[-1000:]}")
        return {}

    # Parse JSON results written by VBench
    scores: Dict[str, float] = {}
    for jf in sorted(Path(output_path).glob("results_*.json")):
        try:
            with open(jf) as fh:
                data = json.load(fh)
            for dim, vals in data.items():
                if isinstance(vals, list) and len(vals) >= 1:
                    scores[dim] = float(vals[0])
                elif isinstance(vals, (int, float)):
                    scores[dim] = float(vals)
        except Exception as e:
            print(f"[VBench] Failed to parse {jf}: {e}")

    print(f"[VBench] Scores: {scores}")
    return scores


# ---------------------------------------------------------------------------
# CLIP score (video ↔ text)
# ---------------------------------------------------------------------------

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
_clip_model_cache: Dict = {}


def load_clip_model(device: Optional[str] = None):
    """
    Load CLIP ViT-L/14 (via HuggingFace transformers).
    Returns (model, processor).  Result is cached per device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device not in _clip_model_cache:
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME).eval().to(device)
        _clip_model_cache[device] = (model, processor)
        print(f"[CLIP] Loaded {_CLIP_MODEL_NAME} on {device}")

    return _clip_model_cache[device]


def compute_clip_score(
    video_path: str,
    prompt: str,
    model=None,
    processor=None,
    device: Optional[str] = None,
    n_frames: int = 8,
) -> float:
    """
    Compute CLIP score for a video and a text prompt.

    Samples *n_frames* uniformly from the full clip, computes per-frame CLIP
    image embeddings, averages + re-normalises them, then returns the cosine
    similarity with the text embedding multiplied by 100 (standard scale).

    Returns a float in roughly [0, 100].
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None or processor is None:
        model, processor = load_clip_model(device)

    frames = extract_frames_from_video(video_path, n=n_frames)

    # Encode images (each frame separately to avoid huge batch)
    img_embeds = []
    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            feat = model.get_image_features(pixel_values=pixel_values)
            feat = F.normalize(feat, dim=-1)
        img_embeds.append(feat)

    avg_img = F.normalize(torch.cat(img_embeds, dim=0).mean(dim=0, keepdim=True), dim=-1)

    # Encode text
    txt_inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
    with torch.no_grad():
        txt_feat = model.get_text_features(**txt_inputs)
        txt_feat = F.normalize(txt_feat, dim=-1)

    score = float((avg_img @ txt_feat.T).item())
    return score * 100

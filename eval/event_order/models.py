"""Frozen models used by the event-order metrics: CLIP ViT-B/32, DINOv2-base, LPIPS (AlexNet), GroundingDINO-base."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F_

CLIP_MODEL = 'openai/clip-vit-base-patch32'
CLIP_REV = 'refs/pr/21'   # safetensors conversion; torch < 2.6 cannot load the .bin
DINO_MODEL = 'facebook/dinov2-base'
GROUNDING_MODEL = 'IDEA-Research/grounding-dino-base'


def read_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


class Scorer:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        import lpips
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REV, use_safetensors=True, torch_dtype=torch.float16).cuda().eval()
        self.cproc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.dino = AutoModel.from_pretrained(DINO_MODEL, torch_dtype=torch.float16).cuda().eval()
        self.dproc = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.lp = lpips.LPIPS(net='alex', verbose=False).cuda().eval()

    @torch.no_grad()
    def clip_sim(self, frames, texts):
        """[n_frames, n_texts] cosine similarity"""
        tx = self.cproc(text=texts, return_tensors='pt', padding=True, truncation=True).to('cuda')
        t = F_.normalize(self.clip.get_text_features(**tx).float(), dim=-1)
        out = []
        for i in range(0, len(frames), 32):
            im = self.cproc(images=frames[i:i + 32], return_tensors='pt').to('cuda')
            out.append(F_.normalize(self.clip.get_image_features(pixel_values=im['pixel_values'].half()).float(), dim=-1) @ t.T)
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def dino_cls(self, frames):
        """[n, 768] L2-normalised DINOv2 CLS embeddings"""
        out = []
        for i in range(0, len(frames), 32):
            im = self.dproc(images=frames[i:i + 32], return_tensors='pt').to('cuda')
            out.append(F_.normalize(self.dino(pixel_values=im['pixel_values'].half()).last_hidden_state[:, 0].float(), dim=-1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def lpips_consecutive(self, frames):
        """[n-1] LPIPS between consecutive frames at 640x352"""
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        x = F_.interpolate(x, size=(352, 640), mode='bilinear', align_corners=False)
        d = []
        for i in range(0, len(x) - 1, 16):
            a, b = x[i:i + 16].cuda(), x[i + 1:i + 17].cuda()
            n = min(len(a), len(b))
            d.append(self.lp(a[:n], b[:n]).flatten().cpu())
        return torch.cat(d).numpy()[:len(frames) - 1]


class Grounder:
    def __init__(self, sc):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(GROUNDING_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).cuda().eval()
        self.sc = sc

    @torch.no_grad()
    def crop_emb(self, frames, subject):
        """DINOv2 embeddings of the best `subject` box per frame (frames without a box are skipped), and their count"""
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        inp = self.proc(images=ims, text=[subject.lower() + '.'] * len(ims), return_tensors='pt').to('cuda')
        res = self.proc.post_process_grounded_object_detection(self.model(**inp), inp.input_ids, threshold=0.3,
                                                               text_threshold=0.25, target_sizes=[im.size[::-1] for im in ims])
        crops = []
        for im, r in zip(ims, res):
            if len(r['scores']):
                x0, y0, x1, y1 = r['boxes'][int(r['scores'].argmax())].tolist()
                if x1 - x0 > 8 and y1 - y0 > 8:
                    crops.append(np.asarray(im.crop((x0, y0, x1, y1))))
        return (self.sc.dino_cls(crops) if crops else None), len(crops)

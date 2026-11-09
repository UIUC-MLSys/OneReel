"""OneReel: prompt-structured scene attention and scene parallelism for multi-shot video DiTs.

Every sentence of a multi-shot prompt is encoded alone and owns one scene: a contiguous run of latent frames. At every
layer and step a scene's video tokens attend to their own frames, `halo` frames of each neighbouring scene and their own
sentence; text tokens attend to their scene's video and their sentence. Optional subject links let a scene also read
the subject tokens of key frames in the other scenes.

  onereel.layout      SceneLayout: scene frame ranges and halo frames
  onereel.attention   varlen attention on FlashAttention-2 or -3
  onereel.parallel    scene parallelism (halo K/V exchange) and Ulysses on xfuser's sequence-parallel group
  onereel.models      HunyuanVideo-1.5, Wan2.2-TI2V-5B, MiniMax-H3
"""
from onereel.layout import SceneLayout

__all__ = ['SceneLayout']

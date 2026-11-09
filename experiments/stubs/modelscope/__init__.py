"""Stand-in for modelscope in the HoloCine runs (experiments/holocine_*.py): diffsynth imports snapshot_download at
module load, but every checkpoint we pass is a local path, so nothing is ever downloaded through it."""


def snapshot_download(*args, **kwargs):
    raise RuntimeError("modelscope download requested: HoloCine checkpoints must be given as local paths")

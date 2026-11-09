"""Multi-GPU execution on xfuser's sequence-parallel group.

  scene    contiguous frame ranges per rank (cost-balanced); a rank receives only the K/V frames of other ranks that its
           scenes and halos read (batch_isend_irecv on contiguous slices) and computes its fully local scenes meanwhile
  ulysses  equal token split; all-to-all of q, k, v to (all tokens, local heads), the same scene attention, all-to-all back
Text tokens are replicated on every rank; their attention outputs are all-reduced over the group.
"""
from onereel.parallel.dist import group, init, peer_rank, rank, world

__all__ = ['group', 'init', 'peer_rank', 'rank', 'world']

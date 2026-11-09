# Changes to xDiT

This directory is [xDiT](https://github.com/xdit-project/xDiT), licensed under Apache-2.0 (`LICENSE.txt`). The OneReel
authors changed the files below; each of them starts with a notice saying so.

| File | Change |
|---|---|
| `xfuser/config/args.py`, `xfuser/config/config.py` | Added and changed |
| `xfuser/core/distributed/parallel_state.py`, `runtime_state.py`, `utils.py` | Changed for the sequence-parallel group |
| `xfuser/model_executor/models/transformers/transformer_wan.py` | Changed |
| `xfuser/model_executor/models/transformers/transformer_minimax_h3.py` | New: MiniMax-H3 transformer with Ulysses sequence parallelism |
| `examples/wan_example.py` | Changed |

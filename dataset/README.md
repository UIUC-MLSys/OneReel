# Dataset inventory

This directory contains both source data and reproducible dataset manifests.
Generated videos and evaluation reports belong under `outputs/` and should not
be added here.

## Construction

| Command | Inputs | Outputs |
|---|---|---|
| `python dataset/build_multishot_bench.py` | `data/external/{holocine,msvbench,shot2story}` (subsets of the public releases) | `multishot_bench.json` |
| `python dataset/add_msvbench.py` | The official MSVBench release (downloaded by the script, fixed revision) | The five MSVBench rows, inserted into `multishot_bench.json` |
| `python dataset/build_minute_bench.py` | Scripts defined in the builder | `minute_bench.json` |

## Checked-in manifests

| Files | Purpose |
|---|---|
| `multishot_bench.json` | Multi-shot test set of the quality evaluation: open-source and our own shot scripts, split into consistency and simple prompts. 35 of the 40 prompts are checked in; `add_msvbench.py` adds the other five. See below. |
| `minute_bench.json` | One-minute multi-scene requests (`experiments/minute_gen.py`). |
| `event_order_bench.json` | Event-order benchmark: does a full-sequence model render prompt sentences in their written order? Default prompt set of `experiments/event_order_attention.py`. See below. |
| `scene_graph_bench.json` | Scene-graph benchmark: four-shot prompts whose shot relations form a graph (`event_order_attention.py --bench`). See below. |
| `timing_bench.json` | Requests of 3 to 12 scenes for the latency figures (`experiments/fig1_latency.py`). |

## Event-order benchmark

16 groups of three visually distinct events, 4 groups per category (`weather`,
`action`, `location`, `object`). Each group is written in the rotations ABC,
BCA and CAB, giving 48 prompts in which every event appears once in every
position, so position and content biases cancel and per-position chance is
exactly 1/3.

- `order_type`: 7 groups have a common-sense order, written as ABC; their BCA
  and CAB prompts violate it (14 `counter` prompts, e.g. noon, night, then
  dawn). This separates following the text order from following common sense.
  The other 9 groups are `neutral`.
- `shot`: `continuous` (same place and subject; weather, action, object) or
  `cut` (the place changes; location).
- Every sentence is self-contained: indefinite articles and no reference to
  another event, so no sentence carries a cue about its own position.
- `subject` names the entity that should look the same in all three events
  (the shared setting in object groups); it is the detector query for
  subject-region consistency.
- `prompt` concatenates the sentences; `prompt_explicit` adds "First, ...
  Then, ... Finally, ...".
- `segments` gives equal latent-frame ranges per sentence for 81 and 117
  frames, both within the native 121 frames (5 s at 24 fps) of
  Wan2.2-TI2V-5B, and the decoded pixel ranges `[start, end)`.
- Groups O3 (unlit vs burning candle) and A3 (reading vs guitar) have the
  most similar sentence texts under CLIP; score them with care.

## Scene-graph benchmark

12 prompts of four self-contained shots, 3 per pattern: `flashback` (A B B'
A'), `crosscut` (A B A' B'), `converge` (A and B apart, then together) and
`linear` (one subject, four actions). `edges` lists every related shot pair
with the subject phrase they share. Unrelated pairs should stay distinct, and
crosscut neighbours are hard cuts. `segments` gives six latent frames per shot
for 93 frames.

## Multi-shot benchmark

40 prompts, 10 per source x category, built by `dataset/build_multishot_bench.py` (its docstring has every rule). The checked-in file omits the five MSVBench stories; run `python dataset/add_msvbench.py` to add them, and do not commit the result.

| Source | consistency (recurring characters, returning settings, several storylines) | simple (one subject, one setting, one continuous activity) |
|---|---|---|
| open | 5 HoloCine inference examples (two characters, shot / reverse shot); 5 MSVBench stories, 8 consecutive shots in which the main character recurs and a setting returns | 10 Shot2Story-20K test clips (real videos, one caption per shot; no on-screen text, titles, broadcasts or animation) |
| ours | 10 scripts: flashback, cross-cutting storylines, a recurring object, one place at different times, two characters converging, one character in many places, shot / reverse shot, pet and owner, a setting that returns, a group of three | 10 scripts: one subject doing one activity in one place |

- Every sentence is self-contained and repeats the full description of each character it shows. HoloCine `[characterN]` and MSVBench character names are replaced by their descriptions; the source text is in `sentences_raw`.
- `lengths` are latent frames per shot (4 pixel frames each). Ours and MSVBench (which has no durations): N(7, 2^2), rounded, clipped to [3, 12], seeded by the prompt id. Shot2Story (real shot lengths) and HoloCine (its cut frames): relative durations kept, rescaled to a mean of 7, minimum 3. `segments[str(frames)]` gives each shot's latent and pixel range.
- `edges` lists shot pairs that share a character, object or (MSVBench) setting.
- Licenses: Shot2Story annotations CC BY-NC-SA 4.0 (non-commercial research; we shortened the openers and normalised the captions, and these rows stay under that license); HoloCine examples from its Apache-2.0 code repository; our scripts CC BY 4.0. The MSVBench release (Hugging Face, MrSunshy/MSVBench) states no license, so its stories are not redistributed here.
- MSVBench sentences open with the same long setting text, so CLIP (77 tokens) barely separates its shots; judge them with a VLM.
- The raw subsets live in `data/external/` (git-ignored); rebuild them from the public releases listed in the builder.

## Naming and retention rules

- Keep immutable source files and the manifests required to reproduce paper
  results under version control.
- Treat caches, ad-hoc API outputs, plots, and one-off score CSVs as generated
  artifacts; they are ignored by `.gitignore`.
- When adding a derived dataset, document its input file and generating command
  in this file.
- Avoid timestamped filenames inside `dataset/`.

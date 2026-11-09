"""Build the multi-shot test set: open-source prompts and our own shot scripts, each split into two categories.

Categories
  consistency  recurring characters / objects, or a setting that returns (A B A), or several storylines: the shots
               must keep identity and scene consistent across cuts
  simple       one subject in one setting doing one continuous activity; the text reads as one coherent paragraph
Sources (10 prompts per source x category)
  open / consistency   5 HoloCine examples (github.com/yihao-meng/HoloCine inference scripts) + 5 MSVBench stories
                       (huggingface.co/datasets/MrSunshy/MSVBench, Dataset/prompt + script), a window of consecutive
                       shots in which the main character recurs and a setting returns
  open / simple        10 Shot2Story-20K test clips (huggingface.co/datasets/mhan/Shot2Story-20K, CC BY-NC-SA 4.0):
                       real multi-shot videos with one caption per shot. Selection: no shot with on-screen text,
                       titles, logos, collages, interfaces, broadcasts or animation; one subject and one setting with
                       continuous activity; every shot a physical scene
  ours / consistency, ours / simple   written below (SCRIPTS)
Normalisation
  Every sentence is self-contained and repeats the full description of each character it shows (entity card).
  HoloCine [characterN] placeholders and MSVBench character names
  are replaced by their descriptions at first mention in each shot; the source text is kept in sentences_raw.
  Shot2Story captions lose their "The video shows" style openers.
Shot durations, in latent frames (4 pixel frames each; pixel range of latent [a, b) is [max(0, 4a - 3), 4b - 3))
  ours and MSVBench   N(7, 2^2) rounded and clipped to [3, 12], seeded by the prompt id (MSVBench has no durations)
  Shot2Story          real shot lengths, HoloCine: its shot_cut_frames; both rescaled so the mean shot is 7 latent
                      frames (relative durations kept), minimum 3
Output: dataset/multishot_bench.json ({"prompts": [...]}) with fields id, source, source_id, license, category,
pattern, subject, entities, sentences, sentences_raw, prompt, lengths, frames, segments, edges, duration_source.

Usage: python dataset/build_multishot_bench.py [--raw data/external] [--out dataset/multishot_bench.json]
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import random
import re
from pathlib import Path

MEAN_LEN, SD_LEN, MIN_LEN, MAX_LEN = 7, 2, 3, 12

# ------------------------------------------------------------------ our scripts
# (id, category, pattern, subject, entities {key: description}, shots [(entity keys shown, text with {key})])
SCRIPTS = [
    ('C01', 'consistency', 'flashback', 'fisherman',
     {'old': 'an old fisherman with a white beard, a deeply weathered face and a yellow raincoat',
      'young': 'a young fisherman with short black hair, a clean-shaven face and a yellow raincoat'},
     [(['old'], '{old} mends a fishing net on a wooden dock at dusk, the harbor lights coming on behind him.'),
      (['young'], '{young} hauls a heavy net onto a small wooden boat on the open sea in bright morning sun.'),
      (['young'], '{young} laughs and holds up a large silver fish on the deck of the small wooden boat.'),
      (['old'], '{old} sets down the net and looks out over the sea from the wooden dock at dusk, smiling quietly.')]),
    ('C02', 'consistency', 'crosscut', 'chef and courier',
     {'chef': 'a chef with a shaved head, a thin black mustache and a white double-breasted jacket',
      'courier': 'a young courier with curly red hair, a green rain jacket and a large orange delivery backpack'},
     [(['chef'], '{chef} tosses noodles in a flaming wok in a busy restaurant kitchen.'),
      (['courier'], '{courier} pedals a bicycle through a rainy city street at night, headlights reflecting on the wet road.'),
      (['chef'], '{chef} packs the noodles into a paper box and seals it at the kitchen pass.'),
      (['courier'], '{courier} stops the bicycle at the back door of the restaurant and wipes rain from her face.'),
      (['chef', 'courier'], '{chef} hands the paper box to {courier} at the back door of the restaurant, both nodding.'),
      (['courier'], '{courier} rides away into the rainy night with the box in the orange backpack.')]),
    ('C03', 'consistency', 'recurring object', 'red umbrella',
     {'umbrella': 'a vintage red umbrella with a curved carved wooden handle',
      'woman': 'a woman with a grey wool coat and a short blonde bob',
      'boy': 'a small boy with a blue school uniform and a yellow backpack'},
     [(['woman', 'umbrella'], '{woman} shakes rain off {umbrella} on a crowded train platform.'),
      (['umbrella'], '{umbrella} lies forgotten on a wooden bench of the empty train platform after the train leaves.'),
      (['boy', 'umbrella'], '{boy} picks up {umbrella} from the wooden bench and looks around for its owner.'),
      (['boy', 'umbrella'], '{boy} walks through heavy rain to a school gate holding {umbrella} over his head.'),
      (['woman', 'boy', 'umbrella'], '{boy} hands {umbrella} back to {woman} outside the school gate, and she smiles.')]),
    ('C04', 'consistency', 'returning location', 'cafe owner',
     {'owner': 'a café owner with a grey bun, round tortoiseshell glasses and a mustard-yellow apron'},
     [(['owner'], '{owner} unlocks the glass door of a small corner café at sunrise and switches on the lights.'),
      (['owner'], '{owner} grinds coffee beans behind the wooden counter of the empty café in the morning.'),
      (['owner'], '{owner} carries two cups to a crowded window table in the busy café at noon.'),
      (['owner'], '{owner} laughs with customers at the wooden counter of the crowded café at noon.'),
      (['owner'], '{owner} wipes the empty window table of the café at night under a single warm lamp.'),
      (['owner'], '{owner} turns the sign on the glass door to closed and locks the café at night.')]),
    ('C05', 'consistency', 'converge', 'detective and informant',
     {'detective': 'a tall detective with a long beige trench coat, a grey fedora and a thick moustache',
      'informant': 'a nervous young man with a shaved head, round glasses and a black leather jacket'},
     [(['detective'], '{detective} reads a crumpled note under a streetlight on a foggy street at night.'),
      (['informant'], '{informant} glances over his shoulder while hurrying through a narrow alley at night.'),
      (['detective'], '{detective} enters a dim jazz bar and sits in a red leather booth.'),
      (['informant'], '{informant} pushes open the door of the dim jazz bar and scans the room.'),
      (['detective', 'informant'], '{informant} slides into the red leather booth across from {detective}.'),
      (['detective', 'informant'], '{informant} passes a small envelope across the table to {detective}, who tucks it into his coat.')]),
    ('C06', 'consistency', 'one character many places', 'traveler',
     {'traveler': 'a woman with a long silver braid, a small scar over her left eyebrow and a teal hooded jacket'},
     [(['traveler'], '{traveler} walks across golden sand dunes under a blazing sun.'),
      (['traveler'], '{traveler} climbs a blue glacier with an ice axe in a snowstorm.'),
      (['traveler'], '{traveler} pushes through giant green leaves in a humid jungle.'),
      (['traveler'], '{traveler} stands on a rocky mountain summit above the clouds at sunrise.'),
      (['traveler'], '{traveler} sits by a small campfire under a starry sky, writing in a notebook.')]),
    ('C07', 'consistency', 'shot reverse shot', 'chess players',
     {'old': 'an old man with a white goatee, a brown flat cap and a green cardigan',
      'girl': 'a girl of about ten with two black pigtails, freckles and a red sweater'},
     [(['old', 'girl'], '{old} and {girl} sit across a wooden chessboard on a park table in autumn.'),
      (['old'], '{old} strokes his goatee and studies the chessboard on the park table.'),
      (['girl'], '{girl} bites her lip and moves a white knight on the chessboard.'),
      (['old'], '{old} raises his eyebrows in surprise at the chessboard.'),
      (['girl'], '{girl} grins and folds her arms at the park table.'),
      (['old', 'girl'], '{old} laughs and shakes hands with {girl} across the chessboard as leaves fall.')]),
    ('C08', 'consistency', 'pet and owner', 'terrier and owner',
     {'dog': 'a small white terrier with one black ear and a red bandana',
      'owner': 'a young man with dreadlocks, a mustard corduroy jacket and white sneakers'},
     [(['dog'], '{dog} waits by the front door of an apartment, wagging its tail.'),
      (['owner', 'dog'], '{owner} clips a leash onto {dog} in the apartment hallway.'),
      (['owner', 'dog'], '{owner} walks {dog} along a tree-lined street in the morning.'),
      (['dog'], '{dog} chases a tennis ball across the green grass of a city park.'),
      (['owner', 'dog'], '{owner} sits on a park bench while {dog} drops the tennis ball at his feet.'),
      (['owner', 'dog'], '{owner} and {dog} walk back into the apartment building at sunset.')]),
    ('C09', 'consistency', 'setting returns', 'violinist',
     {'violinist': 'a young violinist with a black beret, long auburn hair and a long navy coat'},
     [(['violinist'], '{violinist} plays the violin in a busy subway station as commuters rush past.'),
      (['violinist'], '{violinist} packs the violin into its case on the subway platform.'),
      (['violinist'], '{violinist} plays the violin alone on a city rooftop at sunset.'),
      (['violinist'], '{violinist} returns to the busy subway station and opens the violin case on the floor.'),
      (['violinist'], '{violinist} plays again in the subway station while a crowd gathers around her.')]),
    ('C10', 'consistency', 'group', 'three friends',
     {'driver': 'a man with a red beard and a green beanie',
      'map': 'a woman with short pink hair and a denim jacket',
      'guitar': 'a tall man with a black afro and a yellow hoodie'},
     [(['driver', 'map', 'guitar'], '{driver}, {map} and {guitar} ride in an old blue camper van along a coastal highway.'),
      (['map'], '{map} studies a paper map in the passenger seat of the blue camper van.'),
      (['driver'], '{driver} fills the blue camper van with fuel at a desert gas station.'),
      (['guitar'], '{guitar} plays a guitar on the roof of the blue camper van at the gas station.'),
      (['driver', 'map', 'guitar'], '{driver}, {map} and {guitar} run across a beach towards the waves at sunset.'),
      (['driver', 'map', 'guitar'], '{driver}, {map} and {guitar} sit around a bonfire on the beach at night.')]),
    ('S01', 'simple', 'continuous activity', 'potter',
     {'potter': 'a potter with grey dreadlocks tied back and a clay-stained denim apron'},
     [(['potter'], '{potter} slaps a lump of wet clay onto a spinning potter\'s wheel in a sunlit studio.'),
      (['potter'], '{potter} presses both thumbs into the spinning clay to open a bowl.'),
      (['potter'], '{potter} pulls up the wet walls of the bowl between her fingers.'),
      (['potter'], '{potter} smooths the rim of the finished bowl with a wet sponge as the wheel slows down.')]),
    ('S02', 'simple', 'continuous activity', 'surfer',
     {'surfer': 'a surfer with sun-bleached shoulder-length hair and a black wetsuit with a red stripe'},
     [(['surfer'], '{surfer} paddles a white surfboard out through small waves at sunrise.'),
      (['surfer'], '{surfer} turns the board and paddles hard as a large wave rises behind him.'),
      (['surfer'], '{surfer} pops up and rides along the face of the large wave.'),
      (['surfer'], '{surfer} glides out of the breaking wave and raises a fist.')]),
    ('S03', 'simple', 'continuous activity', 'girl',
     {'girl': 'a little girl with curly black hair, a yellow sun hat and a striped swimsuit'},
     [(['girl'], '{girl} fills a red bucket with wet sand on a sunny beach.'),
      (['girl'], '{girl} flips the red bucket over and lifts it to reveal a sand tower.'),
      (['girl'], '{girl} decorates the sand castle with small white shells.'),
      (['girl'], '{girl} claps her hands next to the finished sand castle as a small wave reaches it.')]),
    ('S04', 'simple', 'continuous activity', 'baker',
     {'baker': 'a baker with a round face, flour-dusted forearms, a white cap and a blue checkered apron'},
     [(['baker'], '{baker} kneads a large ball of dough on a floured wooden table in a small bakery.'),
      (['baker'], '{baker} shapes the dough into three round loaves on the wooden table.'),
      (['baker'], '{baker} slashes the tops of the loaves with a small blade.'),
      (['baker'], '{baker} slides the loaves into a glowing brick oven with a long wooden peel.'),
      (['baker'], '{baker} pulls golden crusty loaves out of the brick oven and smiles.')]),
    ('S05', 'simple', 'continuous activity', 'dog',
     {'dog': 'a golden retriever with a green collar'},
     [(['dog'], '{dog} sits on the grass of a sunny park, staring at a thrown tennis ball.'),
      (['dog'], '{dog} sprints across the park lawn after the bouncing tennis ball.'),
      (['dog'], '{dog} leaps and catches the tennis ball in mid-air.'),
      (['dog'], '{dog} trots back across the lawn with the tennis ball in its mouth.')]),
    ('S06', 'simple', 'continuous activity', 'rower',
     {'rower': 'an old man with a white beard, a wide straw hat and a faded red shirt'},
     [(['rower'], '{rower} pushes a wooden canoe off a pebble shore into a misty lake at dawn.'),
      (['rower'], '{rower} paddles the wooden canoe slowly across the still misty lake.'),
      (['rower'], '{rower} rests the paddle across the canoe and watches the sun rise over the lake.')]),
    ('S07', 'simple', 'continuous activity', 'painter',
     {'painter': 'a teenage boy with messy brown hair, paint-splattered overalls and a backwards red cap'},
     [(['painter'], '{painter} dips a wide brush into a can of white paint next to a weathered wooden fence.'),
      (['painter'], '{painter} paints long white strokes along the wooden fence in a sunny backyard.'),
      (['painter'], '{painter} steps back and wipes his forehead in front of the freshly painted white fence.')]),
    ('S08', 'simple', 'continuous activity', 'blacksmith',
     {'blacksmith': 'a muscular blacksmith with a braided brown beard and a scorched leather apron'},
     [(['blacksmith'], '{blacksmith} pulls a glowing orange steel bar out of a coal forge with tongs.'),
      (['blacksmith'], '{blacksmith} hammers the glowing bar on an anvil, sparks flying.'),
      (['blacksmith'], '{blacksmith} turns the bar on the anvil and hammers its edge flat.'),
      (['blacksmith'], '{blacksmith} plunges the hot blade into a bucket of water, steam rising.'),
      (['blacksmith'], '{blacksmith} holds up the dark blade to the light of the forge and inspects it.')]),
    ('S09', 'simple', 'continuous activity', 'dancer',
     {'dancer': 'a street dancer with a high black ponytail, a white crop top and red baggy track pants'},
     [(['dancer'], '{dancer} starts dancing to music on a cardboard sheet in a city plaza.'),
      (['dancer'], '{dancer} spins on her back on the cardboard sheet.'),
      (['dancer'], '{dancer} freezes in a one-handed handstand on the cardboard sheet.'),
      (['dancer'], '{dancer} jumps to her feet and bows to the clapping crowd in the plaza.')]),
    ('S10', 'simple', 'continuous activity', 'mountain biker',
     {'rider': 'a mountain biker with a matte black helmet, orange goggles and a lime green jersey'},
     [(['rider'], '{rider} pushes off at the top of a steep forest trail.'),
      (['rider'], '{rider} rides fast down the dusty forest trail between pine trees.'),
      (['rider'], '{rider} jumps the bike over a fallen log on the forest trail.'),
      (['rider'], '{rider} skids to a stop at the bottom of the trail and lifts the goggles.')]),
]

# ------------------------------------------------------------------ open-source selections
SHOT2STORY = ['hG-BGG2JL4I.1', 'a_HCF3zh1fU.13', 'cwLuQKSSRk0.7', 'HwlztW6zHdY.68', 'hhzOUv4ZGH4.29',
              '-nLK0mxq1UM.20', 'WYsAwHSduBA.1', 'HcT3uyNtPUE.41', 'EBpdXNCqpfs.40', 'cs5bFI9nHlM.22']
SHOT2STORY_SUBJECT = {'hG-BGG2JL4I.1': 'glass bowl', 'a_HCF3zh1fU.13': 'man in red', 'cwLuQKSSRk0.7': 'woman in a red shirt',
                      'HwlztW6zHdY.68': 'bowl of cream', 'hhzOUv4ZGH4.29': 'suitcase', '-nLK0mxq1UM.20': 'bicycle',
                      'WYsAwHSduBA.1': 'boxer in red shorts', 'HcT3uyNtPUE.41': 'toy car', 'EBpdXNCqpfs.40': 'firework',
                      'cs5bFI9nHlM.22': 'race car'}
# MSVBench: story id -> (first shot index, number of shots); windows where the main character recurs and a setting returns
MSVBENCH = {'27': (0, 8), '14': (0, 8), '17': (0, 8), '55': (0, 8), '02': (2, 8)}
# HoloCine examples (order of raw_datasets/holocine/examples.json): characters and setting, hand-parsed
HOLOCINE = [
    ('masquerade', {'character1': 'a mysterious woman with a sleek bob, a sequined silver dress and an ornate feather mask',
                    'character2': 'a dapper gentleman in a black tuxedo with a simple black domino mask'},
     'in a lavish 1920s Art Deco ballroom during a masquerade party'),
    ('painter', {'character1': 'a young painter with paint-smudged cheeks, intense focused eyes and messily tied-up hair'},
     'in a bright, sun-drenched art studio with large windows and canvases'),
    ('boxing gym', {'character1': 'a weary, aging boxing coach with a towel around his neck and a kind but tough face',
                    'character2': 'a young female boxer with a focused, intense expression and wrapped hands'},
     'in a gritty underground boxing gym with punching bags and brick walls'),
    ('celtic ruin', {'character1': 'a skeptical woman historian in practical hiking gear with a camera',
                     'character2': 'the translucent spectral figure of an ancient Celtic queen with long flowing red hair and a silver circlet'},
     'among mossy standing stones of a misty ancient Celtic ruin at dawn'),
    ('elf forest', {'character1': 'an ancient elf with long silver hair braided with glowing flowers and ethereal white robes',
                    'character2': 'a lost human child with short messy brown hair and wide fearful eyes, clutching a wooden toy'},
     'in an enchanted bioluminescent forest at twilight with giant glowing mushrooms'),
]
# per-shot character for pronoun-only HoloCine shots (example index, shot index) -> character key
HOLOCINE_PRONOUN = {(1, 1): 'character1', (1, 2): 'character1', (1, 5): 'character1',
                    (2, 5): 'character2', (3, 1): 'character1'}
HOLOCINE_NO_CHARACTER = {(1, 3)}                      # close-up of the canvas only
OPENERS = re.compile(r'^(the video (shows|presents|begins with)|in the video,?( this is)?|in this video,?|this is a video of|'
                     r'this video shows|it is|it\'s)\s+', re.I)


def normal_lengths(key, k):
    rnd = random.Random(f'multishot-{key}')
    return [min(MAX_LEN, max(MIN_LEN, round(rnd.gauss(MEAN_LEN, SD_LEN)))) for _ in range(k)]


def rescale(durations):
    m = sum(durations) / len(durations)
    return [max(MIN_LEN, round(d / m * MEAN_LEN)) for d in durations]


def segments(lengths):
    L, out, a = sum(lengths), [], 0
    for n in lengths:
        out.append({'latent': [a, a + n], 'pixel': [max(0, 4 * a - 3), 4 * (a + n) - 3]})
        a += n
    frames = 4 * L - 3
    return frames, {str(frames): {'latent_frames': L, 'per_sentence': out}}


def capfirst(s):
    return s[:1].upper() + s[1:]


def clean(s):
    """straight quotes and single spaces (Wan's prompt_clean must leave the text unchanged)"""
    s = s.translate(str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"'}))
    return re.sub(r'\s+', ' ', s).strip()


def row(pid, source, source_id, license_, category, pattern, subject, entities, sentences, raw, lengths, edges, dur_src):
    frames, seg = segments(lengths)
    sentences = [clean(s) for s in sentences]
    return dict(id=pid, source=source, source_id=source_id, license=license_, category=category, pattern=pattern,
                subject=subject, entities=entities, sentences=sentences, sentences_raw=raw, prompt=' '.join(sentences),
                shots=len(sentences), lengths=lengths, frames=frames, segments=seg, edges=edges, duration_source=dur_src)


def ours():
    rows = []
    for pid, cat, pattern, subject, ents, shots in SCRIPTS:
        sents = [capfirst(t.format(**ents)) for _, t in shots]
        keys = [set(k) for k, _ in shots]
        edges = [[i, j] for i, j in itertools.combinations(range(len(shots)), 2) if keys[i] & keys[j]]
        rows.append(row(f'ours_{pid}', 'ours', pid, 'CC BY 4.0 (this work)', cat, pattern, subject,
                        [{'name': k, 'description': v} for k, v in ents.items()], sents, None,
                        normal_lengths(pid, len(shots)), edges, 'normal(7, 2)'))
    return rows


def holocine(raw):
    ex = json.loads((raw / 'holocine' / 'examples.json').read_text())
    rows = []
    for e, (name, chars, setting) in zip(range(len(ex)), HOLOCINE):
        x = ex[e]
        sents, keys = [], []
        for s, cap in enumerate(x['shots']):
            seen, text = set(), cap
            for c, desc in chars.items():
                if f'[{c}]' in text:
                    text = re.sub(rf"\[{c}\]'s (\w+)", lambda m: f'the {m.group(1)} of {desc}', text, count=1)
                    text = text.replace(f'[{c}]', desc, 1).replace(f'[{c}]', 'them')
                    seen.add(c)
            if (e, s) in HOLOCINE_PRONOUN:
                c = HOLOCINE_PRONOUN[(e, s)]
                text = f'{text.rstrip(".")}; the shot shows {chars[c]}.'
                seen.add(c)
            if not seen and (e, s) not in HOLOCINE_NO_CHARACTER:   # two-shots that name nobody show every character
                seen = set(chars)
                text = f'{text.rstrip(".")}; the shot shows ' + ' and '.join(chars.values()) + '.'
            sents.append(f'{text.rstrip(".")}, {setting}.')
            keys.append(seen)
        n = x['num_frames']
        cuts = x['cuts'] or [round(n * i / len(x['shots'])) for i in range(1, len(x['shots']))]
        b = [0] + cuts + [n]
        edges = [[i, j] for i, j in itertools.combinations(range(len(sents)), 2) if keys[i] & keys[j]]
        rows.append(row(f'holo_{e + 1}', 'HoloCine', f'inference example {e + 1} ({name})', 'Apache-2.0 (code repo)',
                        'consistency', 'two characters, one scene' if len(chars) > 1 else 'one character, one scene',
                        list(chars.values())[0].split(' with ')[0].removeprefix('a ').removeprefix('an '),
                        [{'name': k, 'description': v} for k, v in chars.items()], sents, x['shots'],
                        rescale([b[i + 1] - b[i] for i in range(len(sents))]), edges,
                        'HoloCine shot_cut_frames' if x['cuts'] else 'HoloCine even split'))
    return rows


def msvbench(raw):
    rows = []
    for sid, (a, k) in MSVBENCH.items():
        s = json.loads((raw / 'msvbench' / 'Dataset' / 'script' / f'{sid}.json').read_text())
        lines = [l for l in (raw / 'msvbench' / 'Dataset' / 'prompt' / f'{sid}.txt').read_text().split('\n') if l.strip()]
        clips = [(sc['environment'], c) for sc in s['script'] for c in sc['clips']]
        chars = {str(c['id']): c for c in s['characters']}
        sents, raws, keys = [], [], []
        for i in range(a, a + k):
            env, clip = clips[i]
            text = clean(lines[i])
            for cid in clip['characters']:
                c = chars[str(cid)]
                d = c['description'].rstrip('.')
                d = d[0].lower() + d[1:]
                m = re.search(rf"{re.escape(c['name'])}(?!'s)", text)       # first mention that is not possessive
                if m:
                    text = text[:m.end()] + f' ({d})' + text[m.end():]
                elif c['name'] in text:
                    text = f"{c['name']} is {d}. {text}"
            sents.append(text)
            raws.append(lines[i])
            keys.append(set(clip['characters']) | {f'env{env}'})
        main = max(chars.values(), key=lambda c: sum(str(c['id']) in map(str, kk) for kk in keys))
        head = re.split(r'[,.]| with | in his | in her ', main['description'], maxsplit=1)[0]
        subject = re.sub(r'^(a|an|the)\s+', '', head.strip(), flags=re.I).lower()
        edges = [[i, j] for i, j in itertools.combinations(range(k), 2) if keys[i] & keys[j]]
        envs = [clips[i][0] for i in range(a, a + k)]
        rows.append(row(f'msv_{sid}', 'MSVBench', f'story {sid} shots {a + 1}-{a + k}', 'see MSVBench release', 'consistency',
                        f'story, settings {envs}', subject,
                        [{'name': c['name'], 'description': c['description']} for c in chars.values()],
                        sents, raws, normal_lengths(f'msv_{sid}', k), edges, 'normal(7, 2) (no source durations)'))
    return rows


def shot2story(raw):
    data = {x['video'][:-4]: x for x in json.loads((raw / 'shot2story' / '20k_test.json').read_text())}
    meta = {r['clip_id']: r for r in csv.DictReader(open(raw / 'shot2story' / '20k_meta.csv'))}
    rows = []
    for cid in SHOT2STORY:
        x = data[cid]
        caps = ast.literal_eval(x['captions']) if isinstance(x['captions'], str) else x['captions']
        sents = []
        for c in caps:
            c = OPENERS.sub('', c.strip())
            c = re.sub(r'^this is (a )?', '', c, flags=re.I)
            sents.append(capfirst(c))
        shots = [tuple(map(int, n[:-4].rsplit('_', 2)[1:])) for n in x['video_names']]
        edges = [[i, j] for i, j in itertools.combinations(range(len(sents)), 2)]
        rows.append(row(f's2s_{cid}', 'Shot2Story-20K test', cid, 'CC BY-NC-SA 4.0', 'simple', 'continuous activity',
                        SHOT2STORY_SUBJECT[cid], [], sents, caps, rescale([e - s + 1 for s, e in shots]), edges,
                        f"Shot2Story shot frames ({ast.literal_eval(meta[cid]['categories'])[0]})"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, default=Path('data/external'))
    ap.add_argument('--out', type=Path, default=Path('dataset/multishot_bench.json'))
    a = ap.parse_args()
    rows = holocine(a.raw) + msvbench(a.raw) + shot2story(a.raw) + ours()
    a.out.write_text(json.dumps({'prompts': rows}, indent=1, ensure_ascii=False))
    for (src, cat), g in itertools.groupby(sorted(rows, key=lambda r: (r['source'] != 'ours', r['category'])),
                                          key=lambda r: ('ours' if r['source'] == 'ours' else 'open', r['category'])):
        g = list(g)
        lens = [n for r in g for n in r['lengths']]
        print(f'{src:4s} {cat:11s} {len(g):2d} prompts, shots {min(r["shots"] for r in g)}-{max(r["shots"] for r in g)}, '
              f'shot latent frames mean {sum(lens) / len(lens):.1f} (min {min(lens)}, max {max(lens)})')
    print('wrote', a.out)


if __name__ == '__main__':
    main()

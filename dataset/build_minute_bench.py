"""One-minute multi-scene requests for the long-video capability study (HunyuanVideo-1.5 latent frames: 4 pixel frames
each, 24 fps). Writes dataset/minute_bench.json in the multishot_bench.json format (sentences, joint prompt, lengths in
latent frames, frames, segments), so experiments/frame_memory_smoke.py, experiments/minute_gen.py and the eval scripts
read it unchanged (subject: the main character GroundingDINO tracks for identity). Five stories with recurring characters and places, 26-34 scenes; every request has 356 latent
frames (59.2 s), the lighthouse request with the lengths of the first pilot.
Usage: python dataset/build_minute_bench.py
"""
import json
from pathlib import Path

import numpy as np

TOTAL = 356


def lighthouse():
    keeper = 'an elderly lighthouse keeper with short silver hair, round glasses and a navy wool sweater'
    girl = 'a young girl with curly red hair in two braids, a yellow raincoat and green rubber boots'
    tower = 'a red-and-white striped lighthouse on a rocky coast'
    shots = [
        f'Wide shot at dawn of {tower}, waves breaking softly on the rocks below.',
        f'Close-up of {keeper} waking up in a small wooden bedroom, putting on her glasses.',
        f'{keeper} climbs the spiral iron staircase inside {tower}, one hand on the railing.',
        f'{keeper} polishes the huge glass lens at the top of {tower}, morning light flaring through it.',
        f'{girl} runs along a cliff path toward {tower}, a basket swinging in her hand.',
        f'{keeper} opens the wooden door of {tower} and hugs {girl}.',
        f'In a cozy wooden kitchen, {keeper} pours tea while {girl} unpacks bread from the basket.',
        f'Close-up of {girl} laughing at the kitchen table, crumbs on her cheek.',
        f'{keeper} shows {girl} an old logbook full of ink drawings of ships.',
        f'Close-up of a yellowed logbook page with a hand-drawn sailboat named Wren.',
        f'{girl} and {keeper} walk down to a small harbor with fishing boats bobbing in the water.',
        f'{keeper} points at a little blue sailboat named Wren tied to the wooden pier.',
        f'{girl} sits in the blue sailboat Wren as {keeper} unties the rope from the pier.',
        f'Wide shot of the blue sailboat Wren gliding across calm turquoise water, {tower} in the distance.',
        f'{girl} trails her hand in the water over the side of the sailboat, smiling.',
        f'{keeper} looks up at dark clouds gathering on the horizon, her expression turning serious.',
        f'The wind picks up and the sail of the blue sailboat Wren snaps tight as waves grow.',
        f'{keeper} steers the blue sailboat Wren back toward the harbor through choppy water.',
        f'{girl} grips the edge of the boat, rain starting to fall on her yellow raincoat.',
        f'{keeper} and {girl} tie the blue sailboat Wren to the pier as rain pours down.',
        f'Night falls and a storm lashes {tower}, huge waves crashing over the rocks.',
        f'Inside {tower}, {keeper} lights an oil lamp while thunder rumbles outside.',
        f'{girl} watches lightning through a round window of {tower}, wrapped in a blanket.',
        f'{keeper} climbs the spiral iron staircase with the oil lamp, shadows dancing on the walls.',
        f'The great lens at the top of {tower} begins to rotate, its beam sweeping across the stormy sea.',
        f'Far out at sea, a small fishing boat rocks in the storm, turning toward the lighthouse beam.',
        f'{girl} and {keeper} stand together at the window, watching the fishing boat reach the harbor safely.',
        f'Morning after the storm, calm sea and soft pink sky behind {tower}.',
        f'{keeper} draws the blue sailboat Wren into the logbook while {girl} watches over her shoulder.',
        f'Wide shot of {girl} and {keeper} sitting on the rocks below {tower}, looking out at the sunrise.',
    ]
    return dict(id='minute_lighthouse', subject='elderly person with glasses', pattern='two characters, recurring places',
                entities=[dict(name='keeper', description=keeper), dict(name='girl', description=girl)], shots=shots)


def night_market():
    chef = 'a middle-aged street-food chef with a shaved head, a black headband and a white apron'
    boy = 'a teenage apprentice with messy black hair, a red hoodie and a green apron'
    stall = 'a small noodle stall with a striped orange awning and paper lanterns'
    shots = [
        f'Wide shot at dusk of a crowded night market street, {stall} glowing among other stalls.',
        f'{chef} lifts the striped orange awning of {stall} and hangs the paper lanterns.',
        f'{boy} arrives on a bicycle with crates of vegetables strapped to the back.',
        f'{chef} hands {boy} a knife and points at a pile of green onions.',
        f'Close-up of hands of {boy} clumsily chopping green onions on a wooden board.',
        f'{chef} laughs and shows {boy} how to hold the knife properly.',
        f'Close-up of a large wok on a roaring flame, noodles tossed high into the air.',
        f'{chef} flips noodles in the wok with a single swift motion, steam rising around him.',
        f'Customers line up in front of {stall}, holding paper tickets.',
        f'{boy} hands a steaming bowl of noodles to a smiling old woman at {stall}.',
        f'Close-up of a bowl of noodles topped with a soft egg and green onions.',
        f'A sudden gust of wind blows over the paper lanterns of {stall} and one catches fire.',
        f'{boy} grabs a bucket of water and throws it on the burning lantern.',
        f'{chef} pats {boy} on the shoulder, both of them soaked and laughing.',
        f'Rain begins to fall over the night market and people open umbrellas.',
        f'{chef} and {boy} pull the striped orange awning lower as rain drums on it.',
        f'Inside {stall}, {boy} writes a new menu on a chalkboard in neat letters.',
        f'Close-up of the chalkboard menu reading Midnight Noodles with a drawn bowl.',
        f'{chef} tastes a spoonful of broth that {boy} made, his eyebrows rising in surprise.',
        f'{boy} beams with pride, steam from the broth pot curling around him.',
        f'Late at night, the rain stops and the market street shines with reflected neon.',
        f'{chef} counts coins in a tin box under a paper lantern of {stall}.',
        f'{chef} gives {boy} a small brass key on a red string.',
        f'Close-up of the brass key resting in the palm of {boy}.',
        f'At dawn, {boy} alone lifts the striped orange awning of {stall} with the brass key around his neck.',
        f'{chef} watches from across the street with a cup of tea, smiling at {stall}.',
    ]
    return dict(id='minute_night_market', subject='chef with a shaved head', pattern='two characters, one place, day turns',
                entities=[dict(name='chef', description=chef), dict(name='boy', description=boy)], shots=shots)


def mars():
    astro = 'a female astronaut in a white spacesuit with orange shoulder stripes and a gold-tinted visor'
    robot = 'a small four-wheeled rover robot with a round blue camera eye and a dusty solar panel'
    hab = 'a white dome-shaped habitat on red Martian sand'
    shots = [
        f'Wide shot of {hab} at sunrise, a pale blue sun rising over red hills.',
        f'Inside {hab}, {astro} without her helmet drinks coffee by a round window.',
        f'{astro} puts on her helmet, the gold-tinted visor sliding down.',
        f'The airlock door of {hab} opens and red dust swirls in.',
        f'{robot} rolls out of the airlock, its blue camera eye blinking.',
        f'{astro} walks across red Martian sand, {robot} following behind her.',
        f'Close-up of boot prints of {astro} pressed into the red sand.',
        f'{astro} kneels and collects a rock sample with a small scoop.',
        f'{robot} scans the rock sample with a blue light beam.',
        f'Wide shot of {astro} and {robot} climbing a rocky ridge, {hab} small in the distance.',
        f'{astro} looks through binoculars toward a canyon on the horizon.',
        f'A huge red dust storm appears on the horizon, rolling toward them.',
        f'{robot} spins its camera eye toward the storm and beeps urgently.',
        f'{astro} runs back down the ridge, {robot} racing beside her.',
        f'Dust engulfs {astro}, her helmet light cutting through the red haze.',
        f'{robot} stalls in the sand, its wheels spinning helplessly.',
        f'{astro} turns back and lifts {robot} into her arms.',
        f'{astro} trudges through the storm carrying {robot}, barely visible.',
        f'The glowing airlock of {hab} appears through the swirling dust.',
        f'The airlock door of {hab} closes behind {astro} and {robot}.',
        f'Inside {hab}, {astro} removes her helmet, breathing hard, her face streaked with sweat.',
        f'{astro} brushes red dust off the solar panel of {robot} with a soft brush.',
        f'Close-up of the blue camera eye of {robot} flickering back to life.',
        f'{astro} smiles and taps the top of {robot} affectionately.',
        f'Night on Mars, the storm passing, two small moons visible above {hab}.',
        f'Inside {hab}, {astro} records a video log at a console with a blinking red light.',
        f'Close-up of a screen showing a photo of Earth pinned beside the console.',
        f'{robot} rolls over and nudges the leg of {astro}.',
        f'{astro} falls asleep in a chair while {robot} watches beside her.',
        f'Morning, {astro} and {robot} step out of {hab} into clear pink sky.',
        f'{astro} plants a small flag next to the rock sample site.',
        f'{robot} takes a photo of {astro} standing beside the flag.',
        f'Close-up of the photo on the screen of {robot} showing {astro} waving.',
        f'Wide shot of {hab}, {astro} and {robot} under the calm Martian sky.',
    ]
    return dict(id='minute_mars', subject='astronaut', pattern='character and companion, one place',
                entities=[dict(name='astronaut', description=astro), dict(name='robot', description=robot)], shots=shots)


def detective():
    det = 'a tall detective in a gray trench coat and a black fedora with a thin mustache'
    woman = 'a young informant with a short blond bob, a red leather jacket and silver hoop earrings'
    bar = 'a dim jazz bar with green lamps and a long wooden counter'
    shots = [
        'Wide shot of a rainy city street at night, neon signs reflecting on wet pavement.',
        f'{det} stands under a streetlight, rain dripping from the brim of his hat.',
        f'Close-up of {det} reading a crumpled note with an address written on it.',
        f'{det} pushes open the door of {bar}, a saxophone playing inside.',
        f'{woman} sits alone at the long wooden counter of {bar}, stirring a drink.',
        f'{det} sits down next to {woman}, both looking straight ahead.',
        f'Close-up of {woman} sliding a small photograph across the counter.',
        f'Close-up of the photograph showing a man in a white suit beside a black car.',
        f'{det} pockets the photograph and leaves money on the counter of {bar}.',
        f'{woman} glances nervously toward the door of {bar}.',
        'A man in a white suit steps out of a black car in the rain outside.',
        f'{det} watches the man in the white suit through the rain-streaked window of {bar}.',
        f'{det} follows the man in the white suit down a narrow alley.',
        'The man in the white suit disappears through a metal door in the alley.',
        f'{det} picks the lock of the metal door with a thin wire.',
        f'Inside a dusty warehouse, {det} shines a flashlight across stacked crates.',
        f'Close-up of a crate stamped with a red star, {det} prying it open.',
        f'{det} pulls a stack of old documents from the crate, his eyes narrowing.',
        f'Footsteps echo and {det} hides behind the crates, switching off the flashlight.',
        f'{woman} enters the warehouse, looking around in the dark.',
        f'{det} steps out and grabs the arm of {woman}, surprised to see her.',
        f'{woman} whispers urgently to {det}, pointing toward a back exit.',
        f'{det} and {woman} run out of the warehouse into the rain.',
        f'{det} and {woman} hide in a doorway as the black car speeds past.',
        f'Back at {bar}, {det} spreads the documents on a corner table under a green lamp.',
        f'Close-up of {woman} tracing a name on the documents with her finger.',
        f'{det} lights a cigarette and stares at the documents, thinking.',
        f'Dawn breaks over the wet city as {det} and {woman} walk away together down the empty street.',
    ]
    return dict(id='minute_detective', subject='man in a trench coat', pattern='two characters, recurring bar, a third person',
                entities=[dict(name='detective', description=det), dict(name='informant', description=woman)], shots=shots)


def seasons():
    boy = 'a small boy with short brown hair, freckles and a blue striped sweater'
    dog = 'a golden retriever puppy with a red collar'
    oak = 'a large old oak tree on a hill beside a white farmhouse'
    shots = [
        f'Spring, wide shot of {oak}, green leaves and blossoming flowers in the grass.',
        f'{boy} opens a cardboard box and finds {dog} inside.',
        f'Close-up of {dog} licking the face of {boy}, who giggles.',
        f'{boy} and {dog} run across a green meadow toward {oak}.',
        f'{dog} chases a butterfly among yellow flowers.',
        f'{boy} sits under {oak} with {dog} asleep in his lap.',
        f'Spring rain, {boy} and {dog} splash in muddy puddles near the farmhouse.',
        f'{boy} dries {dog} with a towel on the porch of the white farmhouse.',
        f'Summer, {oak} full of dark green leaves under a bright sun.',
        f'{boy} hangs a tire swing from a branch of {oak}.',
        f'{boy} swings on the tire swing while {dog} barks happily below.',
        f'{boy} and {dog} swim in a small pond, water sparkling in the sunlight.',
        f'{dog} shakes water from its fur, spraying {boy} who laughs.',
        f'Summer night, {boy} and {dog} lie in the grass watching fireflies.',
        f'Close-up of fireflies glowing above the sleeping {dog}.',
        f'Autumn, {oak} covered in orange and red leaves.',
        f'{boy} rakes a big pile of orange leaves under {oak}.',
        f'{dog} jumps into the pile of leaves, scattering them everywhere.',
        f'{boy} carves a pumpkin on the porch of the white farmhouse while {dog} watches.',
        f'Close-up of a glowing jack-o-lantern on the porch, {dog} sniffing it.',
        f'Autumn wind blows the last leaves from {oak} as {boy} pulls his sweater tighter.',
        f'Winter, {oak} bare and covered in snow, the white farmhouse with smoke rising from the chimney.',
        f'{boy} in a red scarf builds a snowman under {oak}.',
        f'{dog} steals the carrot nose of the snowman and runs away.',
        f'{boy} chases {dog} through deep snow, both tumbling and laughing.',
        f'Inside the white farmhouse, {boy} and {dog} sit by a crackling fireplace.',
        f'Close-up of {boy} sipping hot chocolate, {dog} resting its head on his knee.',
        f'Snow falls gently outside the window as {boy} falls asleep with {dog}.',
        f'Spring again, {oak} blooming with new green leaves.',
        f'{boy}, a little taller, runs up the hill with {dog}, now grown larger.',
        f'{boy} and {dog} sit together under {oak}, looking at the farmhouse below.',
        f'Wide shot of {oak} on the hill with {boy} and {dog} as the sun sets.',
    ]
    return dict(id='minute_seasons', subject='boy', pattern='boy and dog, one place, four seasons',
                entities=[dict(name='boy', description=boy), dict(name='dog', description=dog)], shots=shots)


def scene_lengths(n, seed):
    """latent frames per scene, about N(TOTAL / n, 2), clipped to [7, 18], summing to TOTAL"""
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(TOTAL / n, 2, n), 7, 18)
    lengths = np.floor(x / x.sum() * TOTAL).astype(int)
    for i in np.argsort(-(x / x.sum() * TOTAL - lengths))[:TOTAL - lengths.sum()]:
        lengths[i] += 1
    return [int(v) for v in lengths]


def request(r, lengths):
    sentences = [s[0].upper() + s[1:] for s in r['shots']]
    total = sum(lengths)
    frames = 4 * (total - 1) + 1
    cuts = np.cumsum([0] + lengths)
    seg = [dict(latent=[int(a), int(b)], pixel=[max(0, 4 * int(a) - 3) if a else 0, 4 * int(b) - 3 if b < total else frames])
           for a, b in zip(cuts[:-1], cuts[1:])]
    return dict(id=r['id'], source='ours', category='consistency', subject=r['subject'], pattern=r['pattern'],
                entities=r['entities'],
                sentences=sentences, prompt=' '.join(sentences), shots=len(sentences), lengths=lengths, frames=frames,
                segments={str(frames): dict(latent_frames=total, per_sentence=seg)})


def main():
    first = lighthouse()
    rng = np.random.default_rng(0)                           # the first pilot's lengths (sum 356)
    rows = [request(first, [int(x) for x in np.clip(np.round(rng.normal(12, 2, len(first['shots']))), 8, 16)])]
    for i, r in enumerate((night_market(), mars(), detective(), seasons()), 1):
        rows.append(request(r, scene_lengths(len(r['shots']), i)))
    out = Path(__file__).resolve().parent / 'minute_bench.json'
    out.write_text(json.dumps(dict(prompts=rows), indent=1))
    for r in rows:
        print(f"{r['id']}: {r['shots']} scenes, {sum(r['lengths'])} latent frames, {r['frames']} frames = "
              f"{r['frames'] / 24:.1f} s, lengths {min(r['lengths'])}-{max(r['lengths'])}")


if __name__ == '__main__':
    main()

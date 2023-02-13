import json 
import os

def normalize_bpm(starting_bpm, bpm_changes, notes):
    '''
    Takes in the starting bpm, a list of bpm changes, and a list of notes. 

    "normalized" bpm is 60, or 1 beat per second. Effectively, the _time 
    field becomes the second at which the note should be hit.

    This is calculated by dividing the _time field by the bpm of the map.

    For each BPM change, we create a "checkpoint" at the time of the BPM change
    which is the time at which the BPM change occurs. 

    For each note, the time at which the note should be hit is calculated as
    (beat of note - beat of checkpoint) * bpm of checkpoint + time of checkpoint. 

    Which is to say, the time of the checkpoint plus the amount of time that has passed 
    since then, which is defined as the number of beats that have passed since then
    multiplied by the bpm of the checkpoint.
    '''
    print(f'args: {starting_bpm}, {bpm_changes}, {len(notes)}')
    pass



def process_map_bpm(map_path):
    '''
    A wrapper around normalize_bpm to handle I/O and different 
    difficulties of the same map.
    '''
    info_file = os.path.join(map_path, 'info.dat')
    with open(info_file) as f:
        info_data = json.load(f)
    bpm = info_data['_beatsPerMinute']

    # Map .dat files end in Standard.dat, OneSaber.dat, etc.
    # For now we'll just process the Standard.dat file. So EasyStandard.dat,
    # ExpertStandard.dat, etc. are the ones we're interested in.

    for file in os.listdir(map_path):
        if not file.endswith('Standard.dat'):
            continue

        with open(os.path.join(map_path, file)) as f:
            map_data = json.load(f)
        
        custom_data = map_data['_customData']
        
        if '_BPMChanges' not in custom_data:
            # No bpm changes, so we can just normalize the bpm directly. 
            bpm_changes = []
        else:
            bpm_changes = custom_data['_BPMChanges']
        
        notes = map_data['_notes']
        normalize_bpm(bpm, bpm_changes, notes)


process_map_bpm('test_map')
process_map_bpm('test_map2')

    
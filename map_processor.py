import json 
import os


def beat_to_timestamp(notes, bpm_changes, original_bpm):
    '''
    Takes in a list of notes, a list of bpm changes, and the original bpm of the map.
    returns the notes, but with the _time field converted to a timestamp in seconds, as
    a float.
    '''
    current_time = 0
    current_bpm = original_bpm
    current_beat = 0

    bpm_change_index = 0
    output_notes = []
    
    for i in range(len(notes)):
        while len(bpm_changes) > bpm_change_index and notes[i]['_time'] >= bpm_changes[bpm_change_index]['_time']:
            current_time += (bpm_changes[bpm_change_index]['_time'] - current_beat) * 60 / current_bpm
            current_beat = bpm_changes[bpm_change_index]['_time']
            current_bpm = bpm_changes[bpm_change_index]['_BPM']

            bpm_change_index += 1

        current_time += (notes[i]['_time'] - current_beat) * 60 / current_bpm
        current_beat = notes[i]['_time']

        output_note = notes[i].copy()
        output_note['_time'] = current_time

        output_notes.append(output_note)

    return output_notes

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
    second_to_timestring = lambda s: f'{int(s // 60):02}:{int(s % 60):02}'
    class Checkpoint:
        def __init__(self, start_beat, end_beat, bpm, start_time):
            # how long this checkpoint lasts
            self.start_beat = start_beat
            self.end_beat = end_beat
            self.num_beats = end_beat - start_beat

            self.bpm = bpm

            # The amount of time that this checkpoint lasts, in seconds.
            self.time = self.num_beats / self.bpm * 60

            self.start_time = start_time

        def __repr__(self):
            # print to a certain precision for readability
            return f'Checkpoint(start_beat={self.start_beat:.2f}, end_beat={self.end_beat:.2f}, bpm={self.bpm:.2f}, start_time={second_to_timestring(self.start_time)}, time={self.time:.2f})'
        
        def get_time(self, beat):
            '''
            Given a beat, returns the time at which that beat should be hit.
            '''
            return (beat - self.start_beat) / self.bpm * 60 + self.start_time
    
    beat_of_last_note = notes[-1]['_time']
    
    # Create a list of checkpoints.
    checkpoints = []
    current_beat = 0
    current_time = 0
    current_bpm = starting_bpm

    for bpm_change in bpm_changes:
        beat = bpm_change['_time']
        checkpoints.append(Checkpoint(current_beat, beat, current_bpm, current_time))
        current_beat = beat

        current_bpm = bpm_change['_BPM']
        current_time = checkpoints[-1].get_time(current_beat)
    
    # Add the final checkpoint.
    checkpoints.append(Checkpoint(current_beat, beat_of_last_note, current_bpm, current_time))

    # checkpoint validation
    start_beat = -1
    for checkpoint in checkpoints:
        print(checkpoint)
        # assert checkpoint.start_beat > start_beat
        # start_beat = checkpoint.start_beat

    # reverse checkpoints so we can pop them off when we're done with them
    checkpoints.reverse()


    new_notes = []
    for note in notes:
        beat = note['_time']
        while checkpoints[-1].end_beat < beat:
            checkpoints.pop()
        
        new_note = note.copy()
        new_note['_time'] = checkpoints[-1].get_time(beat)
        new_notes.append(new_note)
    
    # for note in new_notes:
    #     # convert to readable time
    #     mins = int(note['_time'] // 60)
    #     secs = int(note['_time'] % 60)
    #     # print(f'{mins}:{secs}')
    
    # print(second_to_timestring(new_notes[-1]['_time']))

    
    




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
        
        custom_data = map_data.get('_customData', {})
        
        if '_BPMChanges' not in custom_data:
            # No bpm changes, so we can just normalize the bpm directly. 
            bpm_changes = []
        else:
            bpm_changes = custom_data['_BPMChanges']
        
        notes = map_data['_notes']
        if len(notes) == 0:
            notes= custom_data['_bookmarks']

        notes_with_timestamps = beat_to_timestamp(notes, bpm_changes, bpm)
        seconds_to_timestring = lambda s: f'{int(s // 60):02}:{int(s % 60):02}'

        new_map_data = map_data.copy()
        new_map_data['_notes'] = notes_with_timestamps
        new_map_data.get('_customData', {})['_BPMChanges'] = []

        # Move the old file to an archive folder and write the new file.
        archive_path = os.path.join(map_path, 'archive')
        os.makedirs(archive_path, exist_ok=True)

        f_path = os.path.join(map_path, file)
        f_archive_path = os.path.join(archive_path, file)
        
        # Don't overwrite the archive file if it already exists.
        if not os.path.exists(f_archive_path):
            print(f'Saving old {file} to {f_archive_path}')
            os.rename(f_path, f_archive_path)

        # create and write the new file
        with open(f_path, 'w') as f:
            json.dump(new_map_data, f, indent=4)

        # print the first and last notes
        print(f'First note: {seconds_to_timestring(notes_with_timestamps[0]["_time"])}')
        print(f'Last note: {seconds_to_timestring(notes_with_timestamps[-1]["_time"])}')
    
    # Move the old info.dat file to an archive folder and write the new file.
    info_archive_path = os.path.join(archive_path, 'info.dat')

    if not os.path.exists(info_archive_path):
        print(f'Saving old info.dat to {info_archive_path}')

    info_data['_beatsPerMinute'] = 60

    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=4)
        


# process_map_bpm('test_map')
# process_map_bpm('test_map2')
# process_map_bpm('snowmelt')
# process_map_bpm('excuse_my_rudeness')
process_map_bpm('bassline')

# for file in os.listdir('C:\Program Files\Oculus\Software\Software\hyperbolic-magnetism-beat-saber\Beat Saber_Data\CustomLevels'):
#     print(file)
#     process_map_bpm(os.path.join('C:\Program Files\Oculus\Software\Software\hyperbolic-magnetism-beat-saber\Beat Saber_Data\CustomLevels', file))

    
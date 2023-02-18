import json 
import os

def prettyprint_time(time):
    '''
    Takes in a float representing a time in minutes, and returns a string
    representing the time in minutes and seconds.
    '''
    minutes = int(time)
    seconds = int((time % 1) * 60)
    milliseconds = int((time % 1) * 60000) % 1000
    return f'{minutes}:{seconds:02}:{milliseconds:03}'


def beat_to_timestamp(notes, bpm_changes, original_bpm):
    '''
    Takes in a list of notes, a list of bpm changes, and the original bpm of the map.
    returns the notes, but with the _time field converted to a timestamp in minutes, as
    a float. That makes a bpm of 1 for all output notes.
    '''
    current_time = 0
    current_beat = 0
    current_bpm = original_bpm

    bpm_change_index = 0
    output_notes = []
    
    for i in range(len(notes)):
        while len(bpm_changes) > bpm_change_index and notes[i]['_time'] >= bpm_changes[bpm_change_index]['_time']:
            # Calculate the time at which the bpm change occurs as 
            # print(f'current_time: {cur_time_formatted}, current_beat: {current_beat}, current_bpm: {current_bpm}')
            current_time += (bpm_changes[bpm_change_index]['_time'] - current_beat) / current_bpm
            current_beat = bpm_changes[bpm_change_index]['_time']

            print(f'at time {prettyprint_time(current_time)}, beat {current_beat}, changing from bpm {current_bpm} to {bpm_changes[bpm_change_index]["_BPM"]}')
            current_bpm = bpm_changes[bpm_change_index]['_BPM']
            # print(f'changed bpm to {current_bpm} at beat {current_beat} and current_time {cur_time_formatted}')

            bpm_change_index += 1

        current_time += (notes[i]['_time'] - current_beat) / current_bpm
        current_beat = notes[i]['_time']

        output_note = notes[i].copy()
        output_note['_time'] = current_time

        output_notes.append(output_note)

    return output_notes


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
        seconds_to_timestring = lambda s: f'{int(s)}:{int((s % 1) * 60):02}'

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

    info_data['_beatsPerMinute'] = 1

    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=4)
        

 
def remove_metronome_offset(map_path):
    for file in os.listdir(map_path):
        if not file.endswith('Standard.dat'):
            continue

        with open(os.path.join(map_path, file)) as f:
            map_data = json.load(f)
        
        custom_data = map_data.get('_customData', {})
        
        if '_BPMChanges' not in custom_data:
            return
        else:
            bpm_changes = custom_data['_BPMChanges']
            for bpm_change in bpm_changes:
                del bpm_change['_metronomeOffset']
                del bpm_change['_beatsPerBar']

        new_map_data = map_data.copy()
        new_map_data.get('_customData', {})['_BPMChanges'] = []

        #delete the old file
        os.remove(os.path.join(map_path, file))

        # write it back
        with open(os.path.join(map_path, file), 'w') as f:
            json.dump(new_map_data, f, indent=4)


        
# process_map_bpm('test_map')
# process_map_bpm('test_map2')
# process_map_bpm('snowmelt')
# process_map_bpm('excuse_my_rudeness')
# process_map_bpm('bassline')
remove_metronome_offset('bassline')

# for file in os.listdir('C:\Program Files\Oculus\Software\Software\hyperbolic-magnetism-beat-saber\Beat Saber_Data\CustomLevels'):
#     print(file)
#     process_map_bpm(os.path.join('C:\Program Files\Oculus\Software\Software\hyperbolic-magnetism-beat-saber\Beat Saber_Data\CustomLevels', file))

    
import numpy as np
import torch

def torch2map(matrix_name, bpm, sample_rate, hop_length=128):
    """Creates a map file from a matrix of one-hot note encodings
    with shape (num_frames, 21), the bpm of the song, and the sample rate of the audio.
    """
    torch_tensor = torch.load(matrix_name)
    note_matrix = torch_tensor.cpu().detach().numpy()
    num_frames, num_features = note_matrix.shape
    notes = []
    for frame_idx in range(num_frames):
        if note_matrix[frame_idx, -1] == 1:
            note_time = (frame_idx * hop_length / sample_rate) * (bpm/60)
            line_index = np.argmax(note_matrix[frame_idx, :4])
            line_layer = np.argmax(note_matrix[frame_idx, 4:7])
            note_type = np.argmax(note_matrix[frame_idx, 7:11])
            cut_direction = np.argmax(note_matrix[frame_idx, 11:20])
            notes.append({
                '_time': note_time,
                '_lineIndex': line_index,
                '_lineLayer': line_layer,
                '_type': note_type,
                '_cutDirection': cut_direction
            })
    map_data = {
        '_version': '2.0.0',
        '_notes': notes,
        '_obstacles': [],
        '_events': []
    }
    return map_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, default="test.pt")
    parser.add_argument("-b", "--bpm", type=int, default=1)
    parser.add_argument("-r", "--samplerate", type=int, default=44100)
    args = parser.parse_args()
    torch2map(args.filename, args.bpm, args.samplerate)
    
if __name__ == "__main__":
    main()
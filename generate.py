import argparse, uuid, subprocess
import torch
from .model import MusicTransformer
from .preprocess import SequenceEncoder, apply_sustain
from .helpers import sample, write_midi, vectorize
# import .midi_input
import six
import pretty_midi
from pretty_midi import ControlChange
import yaml
import os, traceback
import copy

class GeneratorError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--model_key", type=str, required=True,
            help="Key in saved_models/model.yaml, helps look up model arguments and path to saved checkpoint.")
    parser.add_argument("--saved_models", type=str, default = 'saved_models',
                        help="path to saved modele where model.yaml resides")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float, 
            default=[1.0],
            help="space-separated list of temperatures to use when sampling")
    parser.add_argument("--n_trials", type=int, default=3,
            help="number of MIDI samples to generate per experiment")
    parser.add_argument("--primer", type=str, default=None, help="Path to the primer")

    parser.add_argument("--play_live", action='store_true', default=False,
            help="play sample(s) at end of script if true")
    parser.add_argument("--keep_ghosts", action='store_true', default=False)
    parser.add_argument("--stuck_note_duration", type=int, default=0)

    args=parser.parse_args()

    model_key = args.model_key

    try:
        yaml_path = os.path.join(args.saved_models, 'model.yaml')
        yaml_data = yaml.safe_load(open(yaml_path))
    except Exception as e:
        raise GeneratorError(f"Could not read yaml..") from e

    try:
        model_dict = yaml_data[model_key]
    except Exception as e:
        raise GeneratorError(f"could not find yaml information for key {model_key}") from e

    model_path = os.path.join(args.saved_models, model_dict["path"])
    model_args = model_dict["args"]
    try:
        state = torch.load(model_path)
    except RuntimeError:
        state = torch.load(model_path, map_location="cpu")
    
    n_velocity_events = 32
    n_time_shift_events = 125

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events,
           min_events=0)

    if args.primer:
        # Read midi primer
        midi_str = six.BytesIO(open(args.primer, 'rb').read())
        p = pretty_midi.PrettyMIDI(midi_str)
        piano_data = p.instruments[0]

        notes = apply_sustain(piano_data)
        note_sequence = sorted(notes, key=lambda x: (x.start, x.pitch))
        ns = vectorize(note_sequence)

        prime_sequence = decoder.encode_sequences([ns])[0]
    else:
        prime_sequence = []

    model = MusicTransformer(**model_args)
    model.load_state_dict(state, strict=False)

    temps = args.temps

    trial_key = str(uuid.uuid4())[:6]
    n_trials = args.n_trials

    keep_ghosts = args.keep_ghosts
    stuck_note_duration = None if args.stuck_note_duration == 0 else args.stuck_note_duration

    for temp in temps:
        print(f"sampling temp={temp}")
        note_sequence = []
        for i in range(n_trials):
            print("generating sequence")
            output_sequence = sample(model, prime_sequence = prime_sequence, sample_length=args.sample_length, temperature=temp)
            note_sequence = decoder.decode_sequence(output_sequence, 
                verbose=True, stuck_note_duration=None)

            output_dir = f"output/{model_key}/{trial_key}/"
            file_name = f"sample{i+1}_{temp}"
            write_midi(note_sequence, output_dir, file_name)

    #for temp in temps:
    #    try:
    #        subprocess.run(['timidity', f"output/{model_key}/{trial_key}/sample{i+1}_{temp}.midi"])
    #    except KeyboardInterrupt:
    #        continue


if __name__ == "__main__":
    main()

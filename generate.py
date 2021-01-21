import argparse, uuid, subprocess
import torch
from model import MusicTransformer
from preprocess import SequenceEncoder
from helpers import sample, write_midi, vectorize
import midi_input
import six
import pretty_midi
from pretty_midi import ControlChange
import yaml
import copy

class GeneratorError(Exception):
    pass

def apply_sustain(piano_data):
    """
    While the sustain pedal is applied during a midi, extend the length of all
    notes to the beginning of the next note of the same pitch or to
    the end of the sustain. Returns a midi notes sequence.
    """
    _SUSTAIN_ON = 0
    _SUSTAIN_OFF = 1
    _NOTE_ON = 2
    _NOTE_OFF = 3

    notes = copy.deepcopy(piano_data.notes)
    control_changes = piano_data.control_changes
    # sequence of SUSTAIN_ON, SUSTAIN_OFF, NOTE_ON, and NOTE_OFF actions
    first_sustain_control = next((c for c in control_changes if c.number == 64),
                                 ControlChange(number=64, value=0, time=0))

    if first_sustain_control.value >= 64:
        sustain_position = _SUSTAIN_ON
    else:
        sustain_position = _SUSTAIN_OFF
    # if for some reason pedal was not touched...
    action_sequence = [(first_sustain_control.time, sustain_position, None)]
    # delete this please
    cleaned_controls = []
    for c in control_changes:
        # Ignoring the sostenuto and damper pedals due to complications
        if sustain_position == _SUSTAIN_ON:
            if c.value >= 64:
                # another SUSTAIN_ON
                continue
            else:
                sustain_position = _SUSTAIN_OFF
        else:
            # look for the next on signal
            if c.value < 64:
                # another SUSTAIN_OFF
                continue
            else:
                sustain_position = _SUSTAIN_ON
        action_sequence.append((c.time, sustain_position, None))
        cleaned_controls.append((c.time, sustain_position))

    action_sequence.extend([(note.start, _NOTE_ON, note) for note in notes])
    action_sequence.extend([(note.end, _NOTE_OFF, note) for note in notes])
    # sort actions by time and type

    action_sequence = sorted(action_sequence, key=lambda x: (x[0], x[1]))
    live_notes = []
    sustain = False
    for action in action_sequence:
        if action[1] == _SUSTAIN_ON:
            sustain = True
        elif action[1] == _SUSTAIN_OFF:
            # find when the sustain pedal is released
            off_time = action[0]
            for note in live_notes:
                if note.end < off_time:
                    # shift the end of the note to when the pedal is released
                    note.end = off_time
                    live_notes.remove(note)
            sustain = False
        elif action[1] == _NOTE_ON:
            current_note = action[2]
            if sustain:
                for note in live_notes:
                    # if there are live notes of the same pitch being held, kill 'em
                    if current_note.pitch == note.pitch:
                        note.end = current_note.start
                        live_notes.remove(note)
            live_notes.append(current_note)
        else:
            if sustain == True:
                continue
            else:
                note = action[2]
                try:
                    live_notes.remove(note)
                except ValueError:
                    print("***Unexpected note sequence...possible duplicate?")
                    pass
    return notes


def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--model_key", type=str, 
            help="Key in saved_models/model.yaml, helps look up model arguments and path to saved checkpoint.")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float, 
            default=[1.0],
            help="space-separated list of temperatures to use when sampling")
    parser.add_argument("--n_trials", type=int, default=3,
            help="number of MIDI samples to generate per experiment")
    parser.add_argument("--live_input", action='store_true', default = False,
            help="if true, take in a seed from a MIDI input controller")

    parser.add_argument("--play_live", action='store_true', default=False,
            help="play sample(s) at end of script if true")
    parser.add_argument("--keep_ghosts", action='store_true', default=False)
    parser.add_argument("--stuck_note_duration", type=int, default=0)

    args=parser.parse_args()

    model_key = args.model_key

    try:
        model_dict = yaml.safe_load(open('saved_models/model.yaml'))[model_key]
    except:
        raise GeneratorError(f"could not find yaml information for key {model_key}")

    model_path = model_dict["path"]
    model_args = model_dict["args"]
    try:
        state = torch.load(model_path)
    except RuntimeError:
        state = torch.load(model_path, map_location="cpu")
    
    n_velocity_events = 32
    n_time_shift_events = 125

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events,
           min_events=0)

    if args.live_input:
        print("Expecting a midi input...")
        #note_sequence = midi_input.read(n_velocity_events, n_time_shift_events)

        # Read midi primer
        midi_str = six.BytesIO(open('primers/cd-cdl-primer.mid', 'rb').read())
        p = pretty_midi.PrettyMIDI(midi_str)
        piano_data = p.instruments[0]
        # We should apply_sustain
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

    for temp in temps:      
        try:
            subprocess.run(['timidity', f"output/{model_key}/{trial_key}/sample{i+1}_{temp}.midi"])
        except KeyboardInterrupt:
            continue

if __name__ == "__main__":
    main()

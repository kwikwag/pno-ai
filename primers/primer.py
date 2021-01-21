
from mido import MidiFile, MidiTrack
import time


in_midi, out_midi = MidiFile('cd-cdl.mid'), MidiFile()
track = MidiTrack()
out_midi.tracks.append(track)

start = time.time()
for i in in_midi.play():
    if time.time() - start >= 10:
        break

    track.append(i)

for t in track:
    #print(t.time * out_midi.ticks_per_beat)
    t.time = int(t.time * out_midi.ticks_per_beat)

out_midi.save('cd-cdl-primer.mid')

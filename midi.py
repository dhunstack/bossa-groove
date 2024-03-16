import pretty_midi
import os
import pickle

midi_quantized_dir = 'midi_quantized/'
midi_humanized_dir = 'midi_humanized/'
midi_humanized_pred_dir = 'midi_humanized_pred/'
os.makedirs(midi_quantized_dir, exist_ok=True)
os.makedirs(midi_humanized_dir, exist_ok=True)
os.makedirs(midi_humanized_pred_dir, exist_ok=True)

# Drum map to midi and order
drum_map = {'KICK': 36,
 'SNARE': 38,
 'HH_CLOSED': 42,
 'HH_OPEN': 46,
 'TOM_3_LO': 43,
 'TOM_2_MID': 47,
 'TOM_1_HI': 50,
 'CRASH': 49,
 'RIDE': 51}
drum_order = ['KICK', 'SNARE', 'HH_CLOSED', 'HH_OPEN', 'TOM_3_LO', 'TOM_2_MID', 'TOM_1_HI', 'CRASH', 'RIDE']

# Given an array of hits for sixteenth notes, convert it into corresponding midi notes using pretty_midi
def hits_to_midi(hits_2bar, filename):
    pm = pretty_midi.PrettyMIDI()
    drum_program = pretty_midi.instrument_name_to_program('Synth Drum')
    drum = pretty_midi.Instrument(program=drum_program, is_drum=True)
    for i, hits_together in enumerate(hits_2bar):
        for j, hit in enumerate(hits_together):
            if hit == 1:
                note = pretty_midi.Note(velocity=40, pitch=drum_map[drum_order[j]], start=i*0.125, end=(i+0.5)*0.125)
                drum.notes.append(note)
    pm.instruments.append(drum)
    pm.write(filename)

# Given an array of hits, velocities, and offsets for sixteenth notes, convert it into corresponding midi notes using pretty_midi
def hits_to_midi_humanized(hits_2bar, velocities_2bar, offsets_2bar, filename):
    pm = pretty_midi.PrettyMIDI()
    drum_program = pretty_midi.instrument_name_to_program('Synth Drum')
    drum = pretty_midi.Instrument(program=drum_program, is_drum=True)
    for i, hits_together in enumerate(hits_2bar):
        for j, hit in enumerate(hits_together):
            if hit == 1:
                note = pretty_midi.Note(velocity=int(velocities_2bar[i][j]*127), pitch=drum_map[drum_order[j]], start=(i+offsets_2bar[i][j])*0.125, end=(i+0.5+offsets_2bar[i][j])*0.125)
                drum.notes.append(note)
    pm.instruments.append(drum)
    pm.write(filename)

# Read the hits, velocities, and offsets from the pickle file
with open('hits_orig.pkl', 'rb') as f:
    hits = pickle.load(f)
with open('velocities_orig.pkl', 'rb') as f:
    velocities = pickle.load(f)
with open('offsets_orig.pkl', 'rb') as f:
    offsets = pickle.load(f)

# Read the predicted velocities and offsets from the pickle file
with open('velocities_pred.pkl', 'rb') as f:
    velocities_pred = pickle.load(f)
    # Set negative values to 0
    velocities_pred[velocities_pred < 0] = 0
    # Set values greater than 1 to 1
    velocities_pred[velocities_pred > 1] = 1

with open('offsets_pred.pkl', 'rb') as f:
    offsets_pred = pickle.load(f)

count=1
for hits_2bar, velocities_2bar, offsets_2bar in zip(hits, velocities, offsets):
    hits_to_midi(hits_2bar, os.path.join(midi_quantized_dir, str(count)+'.mid'))
    hits_to_midi_humanized(hits_2bar, velocities_2bar, offsets_2bar, os.path.join(midi_humanized_dir, str(count)+'.mid'))
    hits_to_midi_humanized(hits_2bar, velocities_pred[count-1], offsets_pred[count-1], os.path.join(midi_humanized_pred_dir, str(count)+'.mid'))
    count+=1


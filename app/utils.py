from music21 import stream

from music21 import converter, instrument, note, chord

def sequence_to_midi(predicted, int_to_note, output_path="generated.mid"):
    """Convierte una secuencia de índices a archivo MIDI."""
    output_notes = []
    offset = 0
    for pattern in predicted:
        symbol = int_to_note[pattern]
        if "." in symbol:
            notes_in_chord = symbol.split(".")
            chord_notes = [note.Note(n) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(symbol)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5  # duración fija

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)


def midi_to_sequence(midi_path):
    """Convierte un archivo MIDI a una lista de notas/códigos."""
    midi = converter.parse(midi_path)

    notes = []
    for element in midi.flatten().notes:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n.pitch) for n in element.notes))
    return notes

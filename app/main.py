# app/main.py
import os
import json
import tempfile
import shutil
from functools import lru_cache
from typing import Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# ML & audio libs
from tensorflow.keras.models import load_model
import music21 as m21
import pretty_midi
import librosa
import matplotlib.pyplot as plt

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_JSON = BASE_DIR / "app/models.json"
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
SOUNDFONT_PATH = os.environ.get("SOUNDFONT_PATH", "")  # e.g. "/usr/share/sounds/sf2/FluidR3_GM.sf2"

for d in (UPLOAD_DIR, GENERATED_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MusicGen API")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utilities for model catalog ----------
def read_models_catalog():
    if not MODELS_JSON.exists():
        return []
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("models", [])

@app.get("/models")
def list_models():
    return {"models": read_models_catalog()}

@app.get("/models/{model_id}")
def get_model_info(model_id: str):
    models = read_models_catalog()
    for m in models:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail="Model not found")

# ---------- Model loading & caching ----------
@lru_cache(maxsize=4)
def load_model_and_vocab(model_path: str, token2idx_path: str):
    # carga keras model + token mappings
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path, compile=False)
    with open(token2idx_path, "r", encoding="utf-8") as f:
        token2idx = json.load(f)
    idx2token = {int(v): k for k, v in token2idx.items()}  # ensure ints for keys
    VOCAB_SIZE = len(token2idx) + 1
    return model, token2idx, idx2token, VOCAB_SIZE

# ---------- Token/sequence helpers (adaptadas a tu notebook) ----------
def sequence_to_tokens(seq_indices, idx2token):
    return [idx2token.get(int(i), "<UNK>") for i in seq_indices]

def build_stream_from_sequence_tokens(seq_tokens):
    """Construye music21.Stream a partir de lista de tokens (C4 o C4.E4.G4)."""
    S = m21.stream.Part()
    S.append(m21.stream.Measure())
    offset = 0
    for t in seq_tokens:
        if "." in t:
            notes = []
            for n in t.split("."):
                try:
                    notes.append(m21.note.Note(int(n)))
                except Exception:
                    notes.append(m21.note.Note(n))
            c = m21.chord.Chord(notes)
            c.quarterLength = 1.0
            c.offset = offset
            S.append(c)
        else:
            try:
                n = m21.note.Note(int(t))
            except Exception:
                n = m21.note.Note(t)
            n.quarterLength = 1.0
            n.offset = offset
            S.append(n)
        offset += 1
    sc = m21.stream.Score()
    sc.insert(0, S)
    return sc

def encode_seed(seed_notes, token2idx):
    print("SEED NOTES: ",seed_notes)
    print("\n\n")
    #print("DICCIONARIO: ",token2idx)
    seed_encoded = []
    not_founded = 0
    for n in seed_notes:
        if n in token2idx:
            seed_encoded.append(token2idx[n])
        else:
            not_founded += 1
    print(f"{not_founded} notes not founded")
    # return python ints
    return seed_encoded, not_founded



def generate_music_auto(model, seed_seq_indices, VOCAB_SIZE, gen_len=200, temperature=1.0):
    SEQ_LEN_INPUT_ACTUAL = model.input_shape[1]
    generated = list(seed_seq_indices)
    for _ in range(gen_len):
        input_seq = np.array([generated[-SEQ_LEN_INPUT_ACTUAL:]])
        preds = model.predict(input_seq, verbose=0)[0]
        preds = np.log(preds + 1e-9) / (temperature + 1e-12)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_idx = int(np.random.choice(range(VOCAB_SIZE), p=preds))
        generated.append(next_idx)
    return generated

# ---------- MIDI <-> tokens ----------
def midi_to_sequence_tokens(midi_path):
    """Convierte MIDI en lista de token strings (C4, C4.E4.G4...)
       Asume resolution: cada cuarto de nota = 1 token en tu dataset."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    # tomaremos nota por 'frame' cuantizado en 1/4 de nota dando resolución 1.0
    # calculamos piano_roll con fs = 1 (column per quarter)
    pr = pm.get_piano_roll(fs=1)  # shape (128, T)
    T = pr.shape[1]
    tokens = []
    for t in range(T):
        pitches = np.where(pr[:, t] > 0)[0]
        if len(pitches) == 0:
            tokens.append("REST")  # si tu vocab no tiene REST, podrías mapear a algo
        elif len(pitches) == 1:
            tokens.append(str(pitches[0]))  # usas ints en tu notebook
        else:
            tokens.append(".".join(str(int(p)) for p in pitches))
    return tokens

# ---------- WAV -> MIDI (heurístico monofónico) ----------
def wav_to_midi(wav_path, midi_out_path):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    # Onset detection + pitch track
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames', backtrack=False)
    times = librosa.frames_to_time(onset_frames, sr=sr)
    # pitch tracking using piptrack
    S = np.abs(librosa.stft(y))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
    # For each time column choose highest magnitude pitch
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    inst = pretty_midi.Instrument(program=piano_program)
    frame_times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
    # get predominant pitch per frame
    last_note = None
    note_on_time = 0.0
    for i in range(pitches.shape[1]):
        col = pitches[:, i]
        mag = magnitudes[:, i]
        if mag.max() < 1e-6:
            cur_pitch = None
        else:
            idx = mag.argmax()
            cur_pitch = col[idx]
            if cur_pitch <= 0: cur_pitch = None
        t = frame_times[i]
        if cur_pitch is not None:
            midi_note = int(np.round(librosa.hz_to_midi(cur_pitch)))
            if last_note is None:
                last_note = midi_note
                note_on_time = t
            elif last_note != midi_note:
                # close previous, start new
                n = pretty_midi.Note(velocity=100, pitch=int(last_note), start=note_on_time, end=t)
                inst.notes.append(n)
                last_note = midi_note
                note_on_time = t
        else:
            if last_note is not None:
                n = pretty_midi.Note(velocity=100, pitch=int(last_note), start=note_on_time, end=t)
                inst.notes.append(n)
                last_note = None
    # close remaining
    if last_note is not None:
        n = pretty_midi.Note(velocity=100, pitch=int(last_note), start=note_on_time, end=frame_times[-1] + 0.1)
        inst.notes.append(n)
    pm.instruments.append(inst)
    pm.write(midi_out_path)
    return midi_out_path

# ---------- MIDI -> WAV (opcional, necesita fluidsynth + soundfont) ----------
def midi_to_wav(midi_path, wav_out_path, soundfont_path=SOUNDFONT_PATH):
    if not soundfont_path or not Path(soundfont_path).exists():
        raise FileNotFoundError("Soundfont not found. Set SOUNDFONT_PATH env var and install fluidsynth.")
    # pretty_midi provides fluidsynth wrapper
    pretty_midi.fluidsynth(str(midi_path), str(wav_out_path), sf2_path=str(soundfont_path))
    return wav_out_path

# ---------- Piano-roll plotting ----------
def save_piano_roll_image(midi_path, img_out_path, fs=10):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    pr = pm.get_piano_roll(fs=fs)
    plt.figure(figsize=(12, 4))
    plt.imshow(pr, aspect='auto', origin='lower')
    plt.xlabel("Frame")
    plt.ylabel("Pitch")
    plt.title(f"Piano Roll: {Path(midi_path).name}")
    plt.tight_layout()
    plt.savefig(img_out_path)
    plt.close()
    return img_out_path

# ---------- Endpoints: upload and generate ----------
class GenerateParams(BaseModel):
    model_id: str
    gen_len: Optional[int] = 50
    temperature: Optional[float] = 1.0
    mode: Optional[str] = "from_song"  # or "from_seed"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Accept midi and audio types
    filename = Path(file.filename).name
    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    # If WAV -> attempt convert to MIDI
    if filename.lower().endswith(".wav") or filename.lower().endswith(".mp3"):
        midi_out = dest.with_suffix(".mid")
        try:
            wav_to_midi(str(dest), str(midi_out))
        except Exception as e:
            return JSONResponse(content={"message": "uploaded but wav->midi conversion failed", "error": str(e)}, status_code=202)
        return {"uploaded": str(dest), "converted_midi": str(midi_out)}
    else:
        return {"uploaded": str(dest)}

@app.post("/generate")
async def generate_audio(
    model_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    gen_len: int = Form(50),
    temperature: float = Form(1.0),
    mode: str = Form("from_song")  # "from_song" or "from_seed"
):
    # find model entry
    models = read_models_catalog()
    entry = next((m for m in models if m["id"] == model_id), None)
    if entry is None:
        raise HTTPException(404, "Model not found")
    model_path = entry["model_path"]
    token2idx_path = entry["token2idx_path"]

    # load model + vocabs
    try:
        model, token2idx, idx2token, VOCAB_SIZE = load_model_and_vocab(model_path, token2idx_path)
    except Exception as e:
        raise HTTPException(500, f"Error loading model/vocab: {e}")

    tmpdir = Path(tempfile.mkdtemp(prefix="musicgen_"))
    try:
        # if user uploaded a file:
        if file is None:
            raise HTTPException(400, "No file provided")
        fname = Path(file.filename).name
        in_path = tmpdir / fname
        with open(in_path, "wb") as f:
            f.write(await file.read())

        # if audio -> convert to midi
        if in_path.suffix.lower() in [".wav", ".mp3", ".flac"]:
            midi_in = tmpdir / (in_path.stem + ".mid")
            try:
                wav_to_midi(str(in_path), str(midi_in))
            except Exception as e:
                raise HTTPException(500, f"WAV->MIDI conversion failed: {e}")
        elif in_path.suffix.lower() in [".mid", ".midi"]:
            midi_in = in_path
        else:
            raise HTTPException(400, "Unsupported file type")

        # tokens from midi
        seed_tokens = midi_to_sequence_tokens(midi_in)
        # depending on mode: if from_song -> take first half as seed and gen_len default to second half length
        if mode == "from_song":
            half = len(seed_tokens) // 2
            seed_slice = seed_tokens[:half]
            default_gen_len = len(seed_tokens) - half
            gen_len = gen_len or default_gen_len
        elif mode == "from_seed":
            seed_slice = seed_tokens
        else:
            raise HTTPException(400, "mode must be 'from_song' or 'from_seed'")

        # encode seed
        seed_indices, not_found = encode_seed(seed_slice, token2idx)
        print(seed_indices)
        if len(seed_indices) == 0:
            raise HTTPException(400, "No seed tokens were encodable to indices.")

        # generate
        gen_indices = generate_music_auto(model, seed_indices, VOCAB_SIZE, gen_len=gen_len, temperature=temperature)
        idx2token_map = {int(k): v for k, v in idx2token.items()}
        gen_tokens = sequence_to_tokens(gen_indices, idx2token_map)

        # build stream, save midi
        gen_stream = build_stream_from_sequence_tokens(gen_tokens)
        midi_out_fp = tmpdir / f"generated_{model_id}.mid"
        gen_stream.write('midi', fp=str(midi_out_fp))

        # attempt to synthesize wav if possible
        wav_out_fp = tmpdir / f"generated_{model_id}.wav"
        wav_created = False
        try:
            midi_to_wav(midi_out_fp, wav_out_fp)
            wav_created = True
        except Exception as e:
            # no fluidsynth / soundfont: ignore but keep midi
            wav_created = False
            synth_error = str(e)

        # piano-roll images: original seed and generated
        seed_midi_fp = tmpdir / "seed.mid"
        # write seed midi for plotting
        seed_stream = build_stream_from_sequence_tokens(seed_slice)
        seed_stream.write('midi', fp=str(seed_midi_fp))

        img_seed = tmpdir / "pianoroll_seed.png"
        img_gen = tmpdir / "pianoroll_gen.png"
        save_piano_roll_image(seed_midi_fp, img_seed)
        save_piano_roll_image(midi_out_fp, img_gen)

        # Move outputs to GENERATED_DIR with stable names for serving
        out_dir = GENERATED_DIR / f"{model_id}_{int(np.random.randint(1e9))}"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(midi_out_fp, out_dir / midi_out_fp.name)
        shutil.copy(img_seed, out_dir / img_seed.name)
        shutil.copy(img_gen, out_dir / img_gen.name)
        if wav_created:
            shutil.copy(wav_out_fp, out_dir / wav_out_fp.name)

        response = {
            "model_id": model_id,
            "mode": mode,
            "gen_len": gen_len,
            "temperature": temperature,
            "tokens_generated_decoded": gen_tokens,
            "tokens_seed": seed_slice,
            "not_found_in_vocab_seed": not_found,
            "files": {
                "midi": str((out_dir / midi_out_fp.name).absolute()),
                "pianoroll_seed": str((out_dir / img_seed.name).absolute()),
                "pianoroll_gen": str((out_dir / img_gen.name).absolute())
            }
        }
        if wav_created:
            response["files"]["wav"] = str((out_dir / wav_out_fp.name).absolute())
        else:
            response["synthesis_note"] = "WAV not created. Install fluidsynth and set SOUNDFONT_PATH env var to enable MIDI->WAV."
            response["synthesis_error"] = synth_error if 'synth_error' in locals() else "unknown"

        return JSONResponse(content=response)

    finally:
        # cleanup tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------- Static file serving helper (dev) ----------
@app.get("/download")
def download(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path=str(p), filename=p.name, media_type="application/octet-stream")

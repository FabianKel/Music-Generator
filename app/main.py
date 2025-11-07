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
from fastapi.staticfiles import StaticFiles


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

# Serve app/data as /static for simple assets (e.g. header video)
static_dir = BASE_DIR / "app" / "data"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Serve the frontend index.html at project root
@app.get("/", include_in_schema=False)
def serve_index():
    idx = BASE_DIR / "app" / "index.html"
    if not idx.exists():
        raise HTTPException(404, "Index file not found")
    return FileResponse(path=str(idx), media_type="text/html")

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
        # Handle special tokens
        if not isinstance(t, str):
            t = str(t)
        tt = t.strip()
        if tt.upper() == "REST" or tt == "<UNK>":
            r = m21.note.Rest()
            r.quarterLength = 1.0
            r.offset = offset
            S.append(r)
        elif "." in tt:
            notes = []
            for n in tt.split("."):
                n = n.strip()
                if not n:
                    continue
                if n.upper() == "REST" or n == "<UNK>":
                    continue
                # try as MIDI number
                try:
                    notes.append(m21.note.Note(int(n)))
                    continue
                except Exception:
                    pass
                # normalize flat notation from 'B-' to 'Bb'
                norm = n.replace('-', 'b')
                try:
                    notes.append(m21.note.Note(norm))
                except Exception:
                    # ignore invalid note
                    continue
            if len(notes) == 0:
                r = m21.note.Rest()
                r.quarterLength = 1.0
                r.offset = offset
                S.append(r)
            else:
                c = m21.chord.Chord(notes)
                c.quarterLength = 1.0
                c.offset = offset
                S.append(c)
        else:
            # single token
            if tt.upper() == "REST" or tt == "<UNK>":
                r = m21.note.Rest()
                r.quarterLength = 1.0
                r.offset = offset
                S.append(r)
            else:
                # try MIDI number first
                try:
                    nobj = m21.note.Note(int(tt))
                except Exception:
                    # normalize flats and try name
                    try:
                        name = tt.replace('-', 'b')
                        nobj = m21.note.Note(name)
                    except Exception:
                        # fallback to rest if cannot parse
                        r = m21.note.Rest()
                        r.quarterLength = 1.0
                        r.offset = offset
                        S.append(r)
                        offset += 1
                        continue
                nobj.quarterLength = 1.0
                nobj.offset = offset
                S.append(nobj)
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
    # helper: try alternative representations
    def try_map(token):
        # direct match
        if token in token2idx:
            return token2idx[token]
        # if token is numeric MIDI like '60' or chord '60.64.67', convert to names and try
        parts = token.split('.')
        converted_parts = []
        all_numeric = True
        for p in parts:
            if p.upper().startswith('REST'):
                converted_parts.append('REST')
                all_numeric = False
                continue
            try:
                n = int(p)
                # use pretty_midi to get name
                try:
                    nm = pretty_midi.note_number_to_name(n)
                except Exception:
                    nm = str(n)
                # normalize flats to token format: pretty_midi uses Bb, token2idx uses B-
                nm = nm.replace('b', '-')
                converted_parts.append(nm)
            except Exception:
                all_numeric = False
                converted_parts.append(p)

        if all_numeric:
            candidate = '.'.join(converted_parts)
            if candidate in token2idx:
                return token2idx[candidate]
            # sometimes token order differs; try sorted by pitch name
            candidate2 = '.'.join(sorted(converted_parts))
            if candidate2 in token2idx:
                return token2idx[candidate2]
        # try token as-is with replacing 'b' -> '-'
        alt = token.replace('b', '-')
        if alt in token2idx:
            return token2idx[alt]
        # fallback to <UNK> if present
        if '<UNK>' in token2idx:
            return token2idx['<UNK>']
        return None

    for n in seed_notes:
        mapped = try_map(n)
        if mapped is not None:
            seed_encoded.append(mapped)
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
            tokens.append("REST")
        elif len(pitches) == 1:
            # convert MIDI number to note name (e.g. 60 -> C4)
            try:
                name = pretty_midi.note_number_to_name(int(pitches[0]))
            except Exception:
                name = str(int(pitches[0]))
            tokens.append(name)
        else:
            names = []
            for p in pitches:
                try:
                    names.append(pretty_midi.note_number_to_name(int(p)))
                except Exception:
                    names.append(str(int(p)))
            tokens.append(".".join(names))
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

        # expose files as relative, safe URLs under /files/{subdir}/{filename}
        files_urls = {
            "midi": f"/files/{out_dir.name}/{midi_out_fp.name}",
            "pianoroll_seed": f"/files/{out_dir.name}/{img_seed.name}",
            "pianoroll_gen": f"/files/{out_dir.name}/{img_gen.name}"
        }
        response = {
            "model_id": model_id,
            "mode": mode,
            "gen_len": gen_len,
            "temperature": temperature,
            "tokens_generated_decoded": gen_tokens,
            "tokens_seed": seed_slice,
            "not_found_in_vocab_seed": not_found,
            "files": files_urls
        }
        if wav_created:
            # expose wav as relative URL under /files
            response["files"]["wav"] = f"/files/{out_dir.name}/{wav_out_fp.name}"
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
    # Secure download: allow only files under GENERATED_DIR
    p = Path(path)
    # if client passed a relative /files/... style path, accept it
    try:
        if str(p).startswith("/files/") or str(p).startswith("files/"):
            # expected format: /files/{subdir}/{filename}
            parts = Path(str(p).lstrip("/"))
            # parts.parts -> ('files','subdir','file')
            if len(parts.parts) < 3:
                raise HTTPException(400, "Invalid file path")
            subdir = parts.parts[1]
            fname = Path("/".join(parts.parts[2:])).name
            resolved = (GENERATED_DIR / subdir / fname).resolve()
        else:
            resolved = p.resolve()
    except Exception:
        raise HTTPException(400, "Invalid path")

    try:
        gen_dir_resolved = GENERATED_DIR.resolve()
        if not resolved.exists() or not resolved.is_file() or not resolved.is_relative_to(gen_dir_resolved):
            raise HTTPException(404, "File not found or not allowed")
    except AttributeError:
        # Fallback for older pathlib without is_relative_to (shouldn't happen on py3.11+)
        try:
            resolved.relative_to(gen_dir_resolved)
        except Exception:
            raise HTTPException(404, "File not found or not allowed")

    return FileResponse(path=str(resolved), filename=resolved.name, media_type="application/octet-stream")


# Serve generated files safely under /files/{subdir}/{filename}
@app.get("/files/{subdir}/{filename}")
def serve_generated_file(subdir: str, filename: str):
    p = (GENERATED_DIR / subdir / filename).resolve()
    gen_dir_resolved = GENERATED_DIR.resolve()
    try:
        if not p.exists() or not p.is_file() or not p.is_relative_to(gen_dir_resolved):
            raise HTTPException(404, "File not found")
    except AttributeError:
        try:
            p.relative_to(gen_dir_resolved)
        except Exception:
            raise HTTPException(404, "File not found")
    return FileResponse(path=str(p), filename=p.name)


@app.get("/synthesize")
def synthesize_midi(path: str):
    """Synthesize a MIDI file to WAV using fluidsynth (requires SOUNDFONT_PATH). Path should be a /files/... URL or filesystem path."""
    # resolve path to file under GENERATED_DIR
    try:
        p = Path(path)
        s = str(path)
        # if the client passed a full URL containing /files/, extract the /files/ path
        if "/files/" in s:
            idx = s.index('/files/')
            s = s[idx:]
        p = Path(s)
        if str(p).startswith("/files/") or str(p).startswith("files/"):
            parts = Path(str(p).lstrip("/"))
            if len(parts.parts) < 3:
                raise HTTPException(400, "Invalid file path")
            subdir = parts.parts[1]
            fname = Path("/".join(parts.parts[2:])).name
            resolved = (GENERATED_DIR / subdir / fname).resolve()
        else:
            resolved = p.resolve()
    except Exception:
        raise HTTPException(400, "Invalid path")

    gen_dir_resolved = GENERATED_DIR.resolve()
    try:
        if not resolved.exists() or not resolved.is_file() or not resolved.is_relative_to(gen_dir_resolved):
            raise HTTPException(404, "File not found or not allowed")
    except AttributeError:
        try:
            resolved.relative_to(gen_dir_resolved)
        except Exception:
            raise HTTPException(404, "File not found or not allowed")

    # ensure it's a midi file
    if resolved.suffix.lower() not in (".mid", ".midi"):
        raise HTTPException(400, "Provided file is not a MIDI file")

    # output wav path in same directory
    wav_out = resolved.with_suffix('.wav')
    try:
        midi_to_wav(resolved, wav_out)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {e}")

    # return the URL to access the wav
    # find subdir name
    subdir_name = resolved.parent.name
    return {"wav": f"/files/{subdir_name}/{wav_out.name}"}


# Diagnostic endpoint: tokenize an uploaded MIDI/WAV and optionally map to a model vocab
@app.post("/tokenize")
async def tokenize_file(file: UploadFile = File(...), model_id: Optional[str] = Form(None)):
    fname = Path(file.filename).name
    tmpdir = Path(tempfile.mkdtemp(prefix="tokenize_"))
    try:
        in_path = tmpdir / fname
        with open(in_path, "wb") as f:
            f.write(await file.read())
        tokens = midi_to_sequence_tokens(in_path)
        resp = {"tokens": tokens}
        if model_id:
            models = read_models_catalog()
            entry = next((m for m in models if m["id"] == model_id), None)
            if entry is None:
                raise HTTPException(404, "Model not found")
            with open(entry["token2idx_path"], "r", encoding="utf-8") as f:
                token2idx = json.load(f)
            mapped = []
            not_found = 0
            for t in tokens:
                m = None
                # reuse encode_seed's mapping logic but for single token
                def try_map_single(token):
                    if token in token2idx:
                        return token2idx[token]
                    parts = token.split('.')
                    converted_parts = []
                    all_numeric = True
                    for p in parts:
                        if p.upper().startswith('REST'):
                            converted_parts.append('REST')
                            all_numeric = False
                            continue
                        try:
                            n = int(p)
                            try:
                                nm = pretty_midi.note_number_to_name(n)
                            except Exception:
                                nm = str(n)
                            nm = nm.replace('b', '-')
                            converted_parts.append(nm)
                        except Exception:
                            all_numeric = False
                            converted_parts.append(p)
                    if all_numeric:
                        cand = '.'.join(converted_parts)
                        if cand in token2idx:
                            return token2idx[cand]
                        cand2 = '.'.join(sorted(converted_parts))
                        if cand2 in token2idx:
                            return token2idx[cand2]
                    alt = token.replace('b', '-')
                    if alt in token2idx:
                        return token2idx[alt]
                    if '<UNK>' in token2idx:
                        return token2idx['<UNK>']
                    return None
                m = try_map_single(t)
                if m is not None:
                    mapped.append(m)
                else:
                    mapped.append(None)
                    not_found += 1
            resp['mapped'] = mapped
            resp['not_found'] = not_found
        return JSONResponse(content=resp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

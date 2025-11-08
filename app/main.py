# app/main.py
import os
import json
import tempfile
import shutil
from functools import lru_cache
from typing import Optional
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import uuid
import statistics
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pretty_midi
import music21 as m21
from tensorflow.keras.models import load_model

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_JSON = BASE_DIR / "app/models.json"
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
SOUNDFONT_PATH = os.environ.get("SOUNDFONT_PATH", "")  

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

#  header video
static_dir = BASE_DIR / "app" / "data"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Ensure non-interactive backend for server environments (avoids Tk errors)
try:
    plt.switch_backend('Agg')
except Exception:
    pass

# Matplotlib styling for clearer, publication-quality plots
try:
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
except Exception:
    pass

try:
    from PIL import Image
    def create_thumbnail_from_large(large_path, thumb_path, size=(360, 240)):
        try:
            with Image.open(large_path) as im:
                im = im.convert('RGB')
                im.thumbnail(size, Image.LANCZOS)
                im.save(thumb_path, optimize=True, quality=85)
                return True
        except Exception:
            return False
except Exception:
    def create_thumbnail_from_large(large_path, thumb_path, size=(360, 240)):
        try:
            shutil.copy(large_path, thumb_path)
            return True
        except Exception:
            return False


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

# ---------- Token/sequence helpers----------
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


# -------------------- XAI endpoints (perturbation-based fallback) --------------------
_XAI_JOBS = {}

def get_xai_dir():
    d = BASE_DIR / "app" / "data" / "xai"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.get("/models/{model_id}/xai")
def list_model_xai(model_id: str):
    d = get_xai_dir()
    patterns = [f for f in d.iterdir() if f.is_file() and f.name.startswith(f"{model_id}_")]
    images = []
    for p in sorted(patterns):
        images.append(f"/static/xai/{p.name}")
    return {"images": images}


def _generate_xai_for_model(model_id: str, kind: str = "perturbation", top_n: int = 12):
    """Generate XAI images for a model and save thumb+large PNGs.
    Attempts SHAP KernelExplainer+summary_plot when requested/available; falls back to perturbation.
    Returns dict with images and token_stats.
    """
    # locate model entry
    models = read_models_catalog()
    entry = next((m for m in models if m["id"] == model_id), None)
    if entry is None:
        raise FileNotFoundError("Model not found")
    model_path = entry["model_path"]
    token2idx_path = entry["token2idx_path"]

    model, token2idx, idx2token, VOCAB_SIZE = load_model_and_vocab(model_path, token2idx_path)

    # choose a reference MIDI (use first available sample in app/data or lullaby)
    sample_midi = BASE_DIR / "app" / "data" / "lullaby-piano.mid"
    if not sample_midi.exists():
        # try twinkle
        sample_midi = BASE_DIR / "app" / "data" / "twinkle-twinkle-little-star.mid"
    if not sample_midi.exists():
        raise FileNotFoundError("No reference MIDI found under app/data for XAI generation")

    tokens = midi_to_sequence_tokens(sample_midi)
    # limit length for speed
    MAX_TOKENS = min(len(tokens), 64)
    tokens = tokens[:MAX_TOKENS]
    seed_indices, not_found = encode_seed(tokens, token2idx)
    if len(seed_indices) == 0:
        raise RuntimeError("Reference MIDI tokens could not be mapped to model vocabulary")

    # prepare base input for prediction: last SEQ_LEN positions expected by model
    SEQ_LEN = model.input_shape[1]
    # ensure we have at least SEQ_LEN by left-padding with first token
    padded = ( [seed_indices[0]] * max(0, SEQ_LEN - len(seed_indices)) ) + seed_indices
    input_seq = np.array([padded[-SEQ_LEN:]])
    baseline_pred = model.predict(input_seq, verbose=0)[0]

    xai_dir = get_xai_dir()
    large_fp = xai_dir / f"{model_id}_{kind}_large.png"
    thumb_fp = xai_dir / f"{model_id}_{kind}_thumb.png"

    # Fast mode: for each unique token in the input (up to top_n), replace all its occurrences
    # a few times and measure the prediction change. Much fewer model calls than per-position sampling.
    if kind == "fast" or kind == "auto":
        rng = np.random.default_rng(2026)
        repeats_per_token = 3
        arr = input_seq[0]
        unique, counts = np.unique(arr, return_counts=True)
        # sort tokens by frequency desc, skip special tokens if needed
        order = np.argsort(-counts)
        token_idxs = [int(unique[i]) for i in order][:top_n]
        token_acc = {}
        for tidx in token_idxs:
            diffs = []
            positions = np.where(arr == tidx)[0]
            if len(positions) == 0:
                continue
            for r in range(repeats_per_token):
                pert = arr.copy()
                for p in positions:
                    pert[p] = int(rng.integers(0, VOCAB_SIZE))
                try:
                    pred = model.predict(np.array([pert]), verbose=0)[0]
                except Exception:
                    continue
                diff = float(np.sum(np.abs(baseline_pred - pred)))
                diffs.append(diff)
            if len(diffs) == 0:
                continue
            token_name = idx2token.get(int(tidx), str(tidx))
            token_acc[token_name] = {"mean": float(statistics.mean(diffs)), "std": float(statistics.pstdev(diffs) if len(diffs) > 1 else 0.0), "samples": len(diffs)}

        stats = [{"token": t, "mean": v["mean"], "std": v["std"], "samples": v["samples"]} for t, v in token_acc.items()]
        stats = sorted(stats, key=lambda x: x["mean"], reverse=True)
        top_stats = stats[:top_n]

        # plotting
        labels = [s["token"] for s in top_stats][::-1]
        means = [s["mean"] for s in top_stats][::-1]
        stds = [s["std"] for s in top_stats][::-1]
        plt.figure(figsize=(9, max(3, 0.4 * len(labels) + 1)))
        y = np.arange(len(labels))
        plt.barh(y, means, xerr=stds, align='center', color='#1b6ca8')
        plt.yticks(y, labels, fontsize=10)
        plt.xlabel('Approx. importance (mean L1 change)', fontsize=11)
        plt.title(f'XAI (fast) for {model_id}', fontsize=12)
        plt.tight_layout()
        plt.savefig(large_fp, dpi=150)
        plt.close()

        # Prefer creating a thumbnail from the large image for consistent appearance
        if not create_thumbnail_from_large(large_fp, thumb_fp):
            try:
                plt.figure(figsize=(4, max(1, 0.28 * len(labels))))
                y = np.arange(len(labels))
                plt.barh(y, means, xerr=stds, align='center', color='#1b6ca8')
                plt.yticks(y, labels, fontsize=8)
                plt.tight_layout()
                plt.savefig(thumb_fp, dpi=100)
                plt.close()
            except Exception:
                try:
                    shutil.copy(large_fp, thumb_fp)
                except Exception:
                    pass

        return {"images": [f"/static/xai/{large_fp.name}", f"/static/xai/{thumb_fp.name}"], "token_stats": top_stats, "method": "fast"}

    # If SHAP requested or auto, try KernelExplainer-based approach first
    tried_shap = False
    if kind in ("shap", "auto"):
        try:
            import shap
            tried_shap = True
            # Build a small set of input rows: the original plus a few perturbed variants
            rng = np.random.default_rng(2025)
            n_variants = 6
            rows = [input_seq[0]]
            for _ in range(n_variants):
                v = list(input_seq[0].copy())
                # apply a few random substitutions
                nsubs = max(1, int(0.12 * len(v)))
                for __ in range(nsubs):
                    pos = int(rng.integers(0, len(v)))
                    v[pos] = int(rng.integers(0, VOCAB_SIZE))
                rows.append(np.array(v))
            rows_arr = np.vstack(rows)

            # explain the probability of the predicted class (argmax)
            target_class = int(np.argmax(baseline_pred))
            def f(x):
                # ensure numpy array
                try:
                    preds = model.predict(np.array(x), verbose=0)
                    return preds[:, target_class]
                except Exception:
                    # fallback: return zeros of appropriate shape
                    arr = np.zeros((np.array(x).shape[0],), dtype=float)
                    return arr

            # background dataset: use the original input as background (fast)
            background = np.array([input_seq[0]])
            explainer = shap.KernelExplainer(f, background)
            nsamples = 40
            shap_vals = explainer.shap_values(rows_arr, nsamples=nsamples)
            # shap_vals shape: (n_rows, SEQ_LEN)
            # aggregate per position across rows
            shap_abs = np.abs(shap_vals)
            pos_means = np.mean(shap_abs, axis=0)
            pos_stds = np.std(shap_abs, axis=0)

            # aggregate by token label (group positions that have same token name)
            token_acc = {}
            pos_tokens = [idx2token.get(int(x), str(x)) for x in input_seq[0]]
            for i, tname in enumerate(pos_tokens):
                token_acc.setdefault(tname, []).append(float(pos_means[i]))

            stats = []
            for t, vals in token_acc.items():
                meanv = statistics.mean(vals) if len(vals) else 0.0
                stdv = statistics.pstdev(vals) if len(vals) else 0.0
                stats.append({"token": t, "mean": meanv, "std": stdv, "samples": len(vals)})
            stats = sorted(stats, key=lambda x: x["mean"], reverse=True)
            top_stats = stats[:top_n]

            # SHAP summary plot (save)
            try:
                plt.figure(figsize=(10, 6))
                # shap.summary_plot expects (n_samples, n_features)
                shap.summary_plot(shap_vals, features=rows_arr, feature_names=pos_tokens, show=False)
                plt.tight_layout()
                plt.savefig(large_fp, dpi=150)
                plt.close()
                # also create thumbnail by copying/resizing
                try:
                    plt.figure(figsize=(4, max(1, 0.25 * len(top_stats))))
                    y = np.arange(len(top_stats[::-1]))
                    labels = [s['token'] for s in top_stats][::-1]
                    means = [s['mean'] for s in top_stats][::-1]
                    stds = [s['std'] for s in top_stats][::-1]
                    plt.barh(y, means, xerr=stds, align='center', color='#264653')
                    plt.yticks(y, labels)
                    plt.tight_layout()
                    # create thumb from large if possible
                    if not create_thumbnail_from_large(large_fp, thumb_fp):
                        try:
                            plt.savefig(thumb_fp, dpi=100)
                        except Exception:
                            try:
                                shutil.copy(large_fp, thumb_fp)
                            except Exception:
                                pass
                    plt.close()
                except Exception:
                    try:
                        shutil.copy(large_fp, thumb_fp)
                    except Exception:
                        pass
            except Exception:
                # If summary_plot fails, fall back to bar chart below
                pass

            return {"images": [f"/static/xai/{large_fp.name}", f"/static/xai/{thumb_fp.name}"], "token_stats": top_stats, "method": "shap"}

        except Exception as e:
            # If SHAP isn't installed or explainer failed, continue to perturbation fallback
            tried_shap = True
            shap_err = str(e)

    # LIME-style local surrogate explainer (faster than SHAP, more detailed than fast heuristic)
    if kind in ("lime", "auto"):
        try:
            # import sklearn on demand
            from sklearn.linear_model import Ridge
            import sklearn
            # parameters for LIME-style perturbations
            N_PERTURBS = 300
            P_KEEP = 0.6  # probability to keep an original token at each position
            kernel_width = max(1.0, SEQ_LEN * 0.25)

            rng = np.random.default_rng(2027)
            X = []  # binary mask: 1 if original token kept, 0 if replaced
            y = []  # model predictions for target_class

            # We'll target the model's predicted class for the original input
            target_class = int(np.argmax(baseline_pred))

            for i in range(N_PERTURBS):
                mask = (rng.random(size=SEQ_LEN) < P_KEEP).astype(int)
                pert = input_seq[0].copy()
                # replace positions where mask==0 with random token
                zeros = np.where(mask == 0)[0]
                for p in zeros:
                    pert[p] = int(rng.integers(0, VOCAB_SIZE))
                try:
                    pred = model.predict(np.array([pert]), verbose=0)[0]
                except Exception:
                    continue
                X.append(mask)
                y.append(float(pred[target_class]))

            if len(y) < 10:
                raise RuntimeError("Not enough perturbation samples for LIME")

            X = np.vstack(X)
            y = np.array(y)

            # compute sample weights using exponential kernel on Hamming distance
            # Hamming distance = number of zeros (differences)
            dists = (SEQ_LEN - X.sum(axis=1)).astype(float)
            weights = np.exp(- (dists ** 2) / (2 * (kernel_width ** 2)))

            # Fit weighted linear model
            model_sur = Ridge(alpha=1.0)
            model_sur.fit(X, y, sample_weight=weights)
            intercept = float(model_sur.intercept_)
            coefs = model_sur.coef_  # per-position contribution when feature==1

            # Aggregate coefficients by token name
            pos_tokens = [idx2token.get(int(x), str(x)) for x in input_seq[0]]
            token_map = {}
            for pos, tname in enumerate(pos_tokens):
                token_map.setdefault(tname, []).append(float(coefs[pos]))

            stats = []
            for t, vals in token_map.items():
                meanv = statistics.mean(vals) if len(vals) else 0.0
                stdv = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                stats.append({"token": t, "mean": meanv, "std": stdv, "samples": len(vals)})
            stats = sorted(stats, key=lambda x: x["mean"], reverse=True)
            top_stats = stats[:top_n]

            # Build waterfall: start at intercept, add contributions for top tokens
            contribs = [s["mean"] for s in top_stats]
            labels = [s["token"] for s in top_stats]
            # compute cumulative positions for waterfall
            cum = intercept
            values = []
            starts = []
            for v in contribs:
                starts.append(cum)
                cum += v
                values.append(v)
            final_pred = float(baseline_pred[target_class])

            # Plot combined: left summary (bar mean±std) and right waterfall
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.2]})
            # left: bar chart of importance
            ax = axes[0]
            y_pos = np.arange(len(labels))[::-1]
            means = [s['mean'] for s in top_stats][::-1]
            stds = [s['std'] for s in top_stats][::-1]
            ax.barh(y_pos, means, xerr=stds, align='center', color='#2b8cbe')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_title('Feature importance (LIME surrogate)')
            ax.set_xlabel('Contribution to target probability')

            # right: waterfall
            ax2 = axes[1]
            # draw bars
            for i, (sval, start) in enumerate(zip(values, starts)):
                if sval >= 0:
                    ax2.barh(i, sval, left=start, color='#2ca02c')
                else:
                    ax2.barh(i, sval, left=start + sval, color='#d62728')
                ax2.text(start + sval + (0.01 if sval >= 0 else -0.01), i, f"{sval:+.3f}", va='center', fontsize=9, color='black')
            ax2.set_yticks(range(len(labels)))
            ax2.set_yticklabels(labels, fontsize=10)
            ax2.axvline(intercept, color='gray', linestyle='--', label='Intercept')
            ax2.axvline(final_pred, color='black', linestyle='-', label='Model pred')
            ax2.set_xlabel('Prediction value')
            ax2.set_title('Waterfall: contributions from intercept to prediction')
            ax2.legend()
            plt.tight_layout()
            plt.savefig(large_fp, dpi=150)
            plt.close()

            # thumbnail
            try:
                fig_thumb, ax_thumb = plt.subplots(1, 1, figsize=(4, max(1, 0.28 * len(labels))))
                y_pos = np.arange(len(labels))[::-1]
                ax_thumb.barh(y_pos, means, xerr=stds, align='center', color='#2b8cbe')
                ax_thumb.set_yticks(y_pos)
                ax_thumb.set_yticklabels(labels, fontsize=8)
                ax_thumb.set_title('Importance (LIME)')
                plt.tight_layout()
                # create thumbnail from the large image for consistent appearance
                if not create_thumbnail_from_large(large_fp, thumb_fp):
                    try:
                        fig_thumb.savefig(thumb_fp, dpi=100)
                    except Exception:
                        try:
                            shutil.copy(large_fp, thumb_fp)
                        except Exception:
                            pass
                plt.close(fig_thumb)
            except Exception:
                try:
                    shutil.copy(large_fp, thumb_fp)
                except Exception:
                    pass

            return {"images": [f"/static/xai/{large_fp.name}", f"/static/xai/{thumb_fp.name}"], "token_stats": top_stats, "method": "lime", "expected_value": intercept, "model_prediction": final_pred}

        except Exception as e:
            lime_err = str(e)
            # fallthrough to next fallback (fast/perturbation)
            pass

    # If SHAP not available or failed, use perturbation fallback
    # perturbation: for each position in the input sequence, sample random substitutions and measure L1 change
    rng = np.random.default_rng(1234)
    samples_per_pos = 18
    token_acc = {}
    for pos in range(len(padded[-SEQ_LEN:])):
        orig = padded[-SEQ_LEN:][pos]
        for s in range(samples_per_pos):
            pert = list(padded[-SEQ_LEN:])
            # replace with a random token index (avoid too many <UNK> if present)
            cand = int(rng.integers(0, VOCAB_SIZE))
            pert[pos] = cand
            pert_arr = np.array([pert])
            try:
                pred = model.predict(pert_arr, verbose=0)[0]
            except Exception:
                continue
            diff = float(np.sum(np.abs(baseline_pred - pred)))
            tname = idx2token.get(int(orig), str(orig))
            token_acc.setdefault(tname, []).append(diff)

    # aggregate stats
    stats = []
    for t, vals in token_acc.items():
        meanv = statistics.mean(vals) if len(vals) else 0.0
        stdv = statistics.pstdev(vals) if len(vals) else 0.0
        stats.append({"token": t, "mean": meanv, "std": stdv, "samples": len(vals)})
    # sort by mean desc
    stats = sorted(stats, key=lambda x: x["mean"], reverse=True)
    top_stats = stats[:top_n]

    labels = [s["token"] for s in top_stats][::-1]
    means = [s["mean"] for s in top_stats][::-1]
    stds = [s["std"] for s in top_stats][::-1]

    plt.figure(figsize=(8, max(3, 0.35 * len(labels) + 1)))
    y = np.arange(len(labels))
    plt.barh(y, means, xerr=stds, align='center', color='#2a9d8f')
    plt.yticks(y, labels)
    plt.xlabel('Perturbation effect (L1 change)')
    plt.title(f'Approximate feature importance (perturbation) for {model_id}')
    plt.tight_layout()
    plt.savefig(large_fp, dpi=150)
    plt.close()

    # thumbnail (smaller)
    # Prefer creating a thumbnail from the large image for consistent appearance
    if not create_thumbnail_from_large(large_fp, thumb_fp):
        try:
            plt.figure(figsize=(4, max(1, 0.25 * len(labels))))
            y = np.arange(len(labels))
            plt.barh(y, means, xerr=stds, align='center', color='#2a9d8f')
            plt.yticks(y, labels)
            plt.xlabel('Effect')
            plt.tight_layout()
            plt.savefig(thumb_fp, dpi=100)
            plt.close()
        except Exception:
            try:
                shutil.copy(large_fp, thumb_fp)
            except Exception:
                pass

    return {
        "images": [f"/static/xai/{large_fp.name}", f"/static/xai/{thumb_fp.name}"],
        "token_stats": top_stats,
        "method": "perturbation",
        ("shap_error" if tried_shap and 'shap_err' in locals() else "note"): (shap_err if tried_shap and 'shap_err' in locals() else ("shap not attempted" if not tried_shap else ""))
    }


@app.post("/models/{model_id}/xai/generate")
async def generate_model_xai(model_id: str, kind: str = Form("fast"), blocking: bool = Form(True), force: bool = Form(False), background_tasks: BackgroundTasks = None):
    """Generate XAI images for a model.
    If blocking=true (default) run synchronously and return images/token_stats.
    If blocking=false, schedule background task and return job_id to poll /xai/jobs/{job_id}.
    """
    # quick cache check: if images already exist and force is False, return cached paths immediately
    xai_dir_quick = get_xai_dir()
    large_quick = xai_dir_quick / f"{model_id}_{kind}_large.png"
    thumb_quick = xai_dir_quick / f"{model_id}_{kind}_thumb.png"
    if large_quick.exists() and thumb_quick.exists() and not force:
        return JSONResponse(content={"images": [f"/static/xai/{large_quick.name}", f"/static/xai/{thumb_quick.name}"], "cached": True})

    if not blocking:
        job_id = str(uuid.uuid4())
        _XAI_JOBS[job_id] = {"status": "pending", "result": None}
        def _bg(job_id_local, mid, k):
            try:
                res = _generate_xai_for_model(mid, kind=k)
                _XAI_JOBS[job_id_local]["status"] = "done"
                _XAI_JOBS[job_id_local]["result"] = res
            except Exception as e:
                _XAI_JOBS[job_id_local]["status"] = "error"
                _XAI_JOBS[job_id_local]["result"] = {"error": str(e)}

        background_tasks.add_task(_bg, job_id, model_id, kind)
        return {"job_id": job_id, "status": "pending"}
    else:
        try:
            res = _generate_xai_for_model(model_id, kind=kind)
            return JSONResponse(content=res)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, f"XAI generation failed: {e}")


@app.get("/xai/jobs/{job_id}")
def get_xai_job(job_id: str):
    if job_id not in _XAI_JOBS:
        raise HTTPException(404, "Job not found")
    return _XAI_JOBS[job_id]

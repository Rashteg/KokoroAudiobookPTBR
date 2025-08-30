# V1.0.5 com uso da GPU
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import os
from pathlib import Path
import glob
import time
import torch  # <-- import adicional

# ─────────────────────────────────────────────────────────────────────────────
# 1) Carrega todos os .txt da pasta "chapters/" em ordem alfabética
# ─────────────────────────────────────────────────────────────────────────────
chapters_folder = Path(__file__).parent / "chapters"
txt_paths = sorted(glob.glob(str(chapters_folder / "*.txt")))
if not txt_paths:
    raise FileNotFoundError(f"Nenhum arquivo .txt encontrado em {chapters_folder}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Lê cada capítulo (tenta UTF-8 e cai em CP1252) e junta tudo num único string
# ─────────────────────────────────────────────────────────────────────────────
texts = []
for txt_file in txt_paths:
    raw = Path(txt_file).read_bytes()
    try:
        content = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        content = raw.decode("cp1252").strip()
    texts.append(content)
master_text = "\n\n".join(texts)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Inicializa a pipeline TTS para pt-br FORÇANDO GPU se disponível
# ─────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
pipeline = KPipeline(lang_code="p", device=device)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Parâmetros de pausa para o marcador "*"
# ─────────────────────────────────────────────────────────────────────────────
sr = 24000
pause_secs = 1.0
silence = np.zeros(int(sr * pause_secs), dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Split por "*" para inserir pausas; gera TTS em cada pedaço
# ─────────────────────────────────────────────────────────────────────────────
segments = master_text.split("*")
buffers = []
total_segments = len(segments)
print(f"Iniciando TTS em {total_segments} segmentos (pausa de {pause_secs}s onde houver '*').\n")

start_time = time.time()
for i, seg in enumerate(segments, start=1):
    seg = seg.strip()
    print(f"[{i}/{total_segments}] Gerando áudio para segmento (primeiros 30 chars):")
    print("   → “" + (seg[:30] + "...") + "”")
    chunk_idx = 0
    for _, _, audio in pipeline(
        seg,
        voice="pm_alex",
        speed=0.77,  # Altere para mudar a velocidade da leitura
        split_pattern=r"\n{2,}"
    ):
        chunk_idx += 1
        print(f"      • chunk #{chunk_idx}  ({len(audio)/sr:.1f}s)")
        buffers.append(audio)
    if i < total_segments:
        buffers.append(silence)
        print(f"      • pausa de {pause_secs}s inserida\n")
    else:
        print()

elapsed = time.time() - start_time
print(f"TTS completo em {elapsed:.1f}s, montando WAV final...\n")

# ─────────────────────────────────────────────────────────────────────────────
# 6) Concatena e salva o WAV final
# ─────────────────────────────────────────────────────────────────────────────
audio_final = np.concatenate(buffers, axis=0)
wav_out = "livro_completo.wav"
sf.write(wav_out, audio_final, sr)

print(f"✅ WAV pronto em: {wav_out}")

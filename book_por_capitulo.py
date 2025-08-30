# V1.0.6 com uso da GPU — um WAV por .txt
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
# 2) Inicializa a pipeline TTS para pt-br FORÇANDO GPU se disponível
# ─────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
pipeline = KPipeline(lang_code="p", device=device)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Parâmetros de pausa para o marcador "*"
# ─────────────────────────────────────────────────────────────────────────────
sr = 24000
pause_secs = 1.0
silence = np.zeros(int(sr * pause_secs), dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Para cada .txt: lê, separa por "*", sintetiza e salva um WAV
# ─────────────────────────────────────────────────────────────────────────────
t0_total = time.time()
total_files = len(txt_paths)

for fidx, txt_file in enumerate(txt_paths, start=1):
    raw = Path(txt_file).read_bytes()
    try:
        content = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        content = raw.decode("cp1252").strip()

    print(f"\n=== [{fidx}/{total_files}] Processando: {Path(txt_file).name} ===")
    segments = content.split("*")
    buffers = []
    total_segments = len(segments)
    print(f"Iniciando TTS em {total_segments} segmentos (pausa de {pause_secs}s onde houver '*').\n")

    t_start = time.time()
    for i, seg in enumerate(segments, start=1):
        seg = seg.strip()
        print(f"[{i}/{total_segments}] Gerando áudio para segmento (primeiros 30 chars):")
        print("   → “" + (seg[:30] + "...") + "”")

        if seg:
            chunk_idx = 0
            for _, _, audio in pipeline(
                seg,
                voice="pm_alex",
                speed=0.77,              # mesma velocidade do V1.0.5
                split_pattern=r"\n{2,}"  # divide por parágrafos (linhas em branco)
            ):
                chunk_idx += 1
                print(f"      • chunk #{chunk_idx}  ({len(audio)/sr:.1f}s)")
                buffers.append(audio)
        else:
            print("      • segmento vazio (sem áudio)")

        if i < total_segments:
            buffers.append(silence)
            print(f"      • pausa de {pause_secs}s inserida\n")
        else:
            print()

    elapsed = time.time() - t_start
    print(f"TTS do arquivo concluído em {elapsed:.1f}s, montando WAV...\n")

    # Concatena e salva o WAV deste .txt
    audio_final = np.concatenate(buffers, axis=0) if buffers else np.zeros(0, dtype=np.float32)
    wav_out = str(Path(txt_file).with_suffix(".wav"))  # salva ao lado do .txt
    sf.write(wav_out, audio_final, sr)

    print(f"✅ WAV pronto: {wav_out} ({len(audio_final)/sr:.1f}s)")

print(f"\n🏁 Finalizado. Arquivos processados: {total_files} | Tempo total: {time.time()-t0_total:.1f}s")

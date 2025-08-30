import re, zipfile, sys
from pathlib import Path

def split_por_asterisco(caminho_txt, out_dir=None):
    src = Path(caminho_txt)
    # Tenta UTF-8 e cai para Latin-1 se precisar
    try:
        texto = src.read_text(encoding='utf-8')
        enc_used = 'utf-8'
    except UnicodeDecodeError:
        texto = src.read_text(encoding='latin-1')
        enc_used = 'latin-1'

    out = Path(out_dir or (src.parent / (src.stem + "_capitulos_por_asterisco")))
    out.mkdir(exist_ok=True, parents=True)

    # Linha inteira envolta por * ... * (sem **)
    pat = re.compile(r"(?m)^[ \t]*\*(?!\*)(.+?)(?<!\*)\*[ \t]*$")

    matches = list(pat.finditer(texto))
    if not matches:
        print(f"Nenhum capítulo encontrado com padrão *...* em: {src.name}")
        return

    # Materiais antes do primeiro capítulo
    if matches[0].start() > 0:
        front = texto[:matches[0].start()]
        if front.strip():
            (out / "00 - Materiais iniciais (antes do primeiro capitulo).txt").write_text(front, encoding=enc_used)

    # Fatia e salva capítulos sem alterar conteúdo
    for i, m in enumerate(matches):
        ini = m.start()
        fim = matches[i+1].start() if i+1 < len(matches) else len(texto)
        titulo = m.group(1).strip()
        safe = re.sub(r"[\\/:*?\"<>|]+", "-", titulo)
        safe = re.sub(r"\s+", " ", safe)[:70].strip()
        num = f"{i+1:02d}" if len(matches) < 100 else f"{i+1:03d}"
        (out / f"{num} - {safe}.txt").write_text(texto[ini:fim], encoding=enc_used)

    # ZIP com todos os capítulos
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, arcname=p.name)

    print(f"Capítulos gerados: {len(matches)} | Encoding: {enc_used}")
    print("Pasta:", out)
    print("ZIP:", out.with_suffix('.zip'))

def escolher_txt(pasta='.'):
    folder = Path(pasta)
    arquivos = sorted([p for p in folder.glob("*.txt") if p.is_file()])
    if not arquivos:
        print("Nenhum .txt foi encontrado nesta pasta.")
        return None

    print("\nArquivos .txt encontrados:\n")
    for i, p in enumerate(arquivos, 1):
        print(f"{i:2d}) {p.name}")
    escolha = input("\nDigite o número do arquivo desejado (Enter = 1, 0 = cancelar): ").strip()

    if escolha == "":
        idx = 1
    elif escolha.isdigit():
        idx = int(escolha)
    else:
        print("Entrada inválida.")
        return None

    if idx == 0:
        return None
    if 1 <= idx <= len(arquivos):
        return arquivos[idx-1]

    print("Número fora do intervalo.")
    return None

if __name__ == "__main__":
    alvo = None
    if len(sys.argv) > 1:
        alvo = Path(sys.argv[1])
    if not alvo:
        alvo = escolher_txt(".")
    if alvo:
        split_por_asterisco(alvo)

"""Roda as Fases 5 e 6 numa GPU local (testado para RTX 4060 de 8 GB, Windows 11), sem supervisao.

Ordem (prioridade): checagens -> escolha do micro-lote -> mid-training M -> SFT -> avaliacoes
(pre-treino, M, SFT e, se o checkpoint existir, C) -> comparacoes (McNemar).

- Retoma de onde parou: rode o mesmo comando de novo (treinos retomam do ultimo checkpoint e
  etapas concluidas sao puladas).
- Para no primeiro erro e registra tudo em logs/local_run.log e logs/local_status.txt.
- Impede o Windows de suspender enquanto roda (a tela pode apagar).

Uso, da raiz do repositorio:
    uv run python scripts/run_local.py            # tudo
    uv run python scripts/run_local.py --check    # so as checagens e o micro-lote (~5 min)

Os dados ficam em ../shared (ao lado do repositorio), com a mesma estrutura do servidor:
shared/data/{mid_m,mid_c,sft,pretrain/val} e shared/checkpoints/pretrain/final.pt.
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
LOG = LOGS / "local_run.log"
STATUS = LOGS / "local_status.txt"
AUTO = ROOT / "configs" / "local" / "_auto"
RESULTS = ROOT / "results" / "final"
EFFECTIVE_BATCH = 24
CANDIDATES = [8, 6, 4, 3, 2, 1]  # divisores de 24


def log(message: str) -> None:
    line = f"[{datetime.now():%d/%m %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def status(stage: str, state: str) -> None:
    with STATUS.open("a", encoding="utf-8") as file:
        file.write(f"{datetime.now():%d/%m %H:%M:%S} | {stage:<28} | {state}\n")


def keep_awake(on: bool) -> None:
    """Windows: impede a suspensao enquanto o processo roda (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def run(stage: str, command: list[str], env: dict) -> None:
    """Roda um comando, espelhando a saida no log. Qualquer erro para tudo."""
    log(f"=== {stage}: {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(command))}")
    status(stage, "iniciado")
    started = time.time()
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        with LOG.open("a", encoding="utf-8") as file:
            for line in process.stdout:
                sys.stdout.write(line)
                file.write(line)
                file.flush()
        code = process.wait()
    hours = (time.time() - started) / 3600
    if code != 0:
        status(stage, f"ERRO (codigo {code}) apos {hours:.2f}h")
        log(f"ERRO em '{stage}' (codigo {code}). Parando. Rode o mesmo comando de novo depois de corrigir: "
            f"as etapas concluidas sao puladas e os treinos retomam do ultimo checkpoint.")
        sys.exit(code)
    status(stage, f"concluido em {hours:.2f}h")


def preflight(shared: Path) -> None:
    import torch

    log(f"repositorio: {ROOT}")
    log(f"dados (shared): {shared}")
    required = {
        "checkpoints/pretrain/final.pt": None,
        "data/mid_m/train/tokens.bin": None, "data/mid_m/val/tokens.bin": None,
        "data/sft/train/tokens.bin": None, "data/sft/train/mask.bin": None,
        "data/sft/val/tokens.bin": None, "data/sft/val/mask.bin": None,
        "data/pretrain/val/tokens.bin": None,
    }
    missing = [name for name in required if not (shared / name).exists()]
    if missing:
        raise SystemExit(f"faltam arquivos em {shared}: {missing}")
    for name in ("data/mid_m/train", "data/sft/train"):
        meta = json.loads((shared / name / "meta.json").read_text())
        expected = meta["num_records"] * (meta["seq_len"] + 1) * 2
        actual = (shared / name / "tokens.bin").stat().st_size
        if actual != expected:
            raise SystemExit(f"{name}/tokens.bin com {actual} bytes; esperado {expected}")
    log("dados: ok (tamanhos conferidos com os meta.json)")

    free_gb = shutil.disk_usage(shared).free / 1e9
    log(f"disco livre: {free_gb:.1f} GB")
    if free_gb < 12:
        raise SystemExit("menos de 12 GB livres no disco dos dados: libere espaco antes de rodar")

    if not torch.cuda.is_available():
        raise SystemExit("o PyTorch nao enxerga a GPU (torch.cuda.is_available() = False)")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("a GPU nao suporta bf16, exigido pelo treino")
    free, total = torch.cuda.mem_get_info()
    log(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | memoria livre "
        f"{free / 2**30:.1f} de {total / 2**30:.1f} GiB")


def probe() -> None:
    """Escolhe o maior micro-lote que cabe com folga (subprocesso: a memoria e devolvida ao sair)."""
    import torch
    import torch.nn.functional as F

    from slm.model import SLM, ModelConfig

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    free, total = torch.cuda.mem_get_info()
    # No Windows, o driver pode usar a RAM quando a GPU enche, sem erro e muito mais devagar.
    # Por isso o limite e pela memoria livre, com folga, e nao por "deu erro de memoria".
    limit = min(0.85 * total, free - 768 * 2**20)
    torch.manual_seed(0)
    model = SLM(ModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
    results = []
    for micro in CANDIDATES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            tokens = torch.randint(0, 50257, (micro, 1025), device=device)
            times = []
            for _ in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(tokens[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1),
                                           reduction="sum") / tokens[:, 1:].numel()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            peak = torch.cuda.max_memory_reserved()
            speed = micro * 1024 / min(times[1:])
            results.append({"micro": micro, "peak_gib": peak / 2**30, "tok_s": speed, "fits": peak <= limit})
        except torch.OutOfMemoryError:
            results.append({"micro": micro, "peak_gib": None, "tok_s": None, "fits": False})
        finally:
            tokens = logits = loss = None
    print(json.dumps({"limit_gib": limit / 2**30, "results": results}))


def choose_micro(env: dict) -> dict:
    state_file = AUTO / "micro.json"
    if state_file.exists():
        choice = json.loads(state_file.read_text())
        log(f"micro-lote ja escolhido antes: {choice['micro']} x {choice['grad_accum']} (de {state_file.name})")
        return choice
    log("medindo a memoria por micro-lote (~2-5 min)...")
    out = subprocess.run([sys.executable, __file__, "--probe"], cwd=ROOT, env=env, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        log(out.stdout + out.stderr)
        raise SystemExit("a medicao de memoria falhou (ver acima)")
    data = json.loads(out.stdout.strip().splitlines()[-1])
    for r in data["results"]:
        peak = "estourou" if r["peak_gib"] is None else f"{r['peak_gib']:.2f} GiB"
        speed = "-" if r["tok_s"] is None else f"{r['tok_s']:,.0f} tok/s"
        log(f"  micro-lote {r['micro']}: pico {peak}, {speed} {'(cabe)' if r['fits'] else '(nao cabe)'}")
    fitting = [r for r in data["results"] if r["fits"]]
    if not fitting:
        raise SystemExit(f"nenhum micro-lote cabe no limite de {data['limit_gib']:.2f} GiB")
    best = max(fitting, key=lambda r: r["micro"])
    choice = {"micro": best["micro"], "grad_accum": EFFECTIVE_BATCH // best["micro"], "tok_s": best["tok_s"]}
    state_file.write_text(json.dumps(choice))
    hours = 374_155_264 / best["tok_s"] / 3600
    log(f"escolhido: micro-lote {choice['micro']} x acumulo {choice['grad_accum']} = {EFFECTIVE_BATCH} "
        f"| ~{best['tok_s']:,.0f} tok/s | estimativa do M: ~{hours:.1f}h de computacao (sem avaliacoes)")
    return choice


def auto_config(name: str, choice: dict) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "local" / f"{name}.yaml").read_text(encoding="utf-8"))
    cfg["batch_size"] = choice["micro"]
    cfg["grad_accum"] = choice["grad_accum"]
    path = AUTO / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shared", type=Path, default=ROOT.parent / "shared")
    parser.add_argument("--check", action="store_true", help="so checagens e escolha do micro-lote")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        probe()
        return

    LOGS.mkdir(exist_ok=True)
    AUTO.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    shared = args.shared.resolve()
    env = {**os.environ, "SLM_SHARED": str(shared), "SLM_SCRATCH": str(shared.parent / "local-cache"),
           "HF_HOME": str(shared.parent / "local-cache" / "hf"), "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}
    keep_awake(True)
    try:
        log("==================== inicio ====================")
        status("checagens", "iniciado")
        preflight(shared)
        choice = choose_micro(env)
        status("checagens", f"ok (micro-lote {choice['micro']} x {choice['grad_accum']})")
        if args.check:
            log("--check: checagens concluidas, nada foi treinado")
            return

        py = sys.executable
        ck = shared / "checkpoints"
        run("mid-training M", [py, "scripts/midtrain.py", "--config", str(auto_config("mid_m", choice))], env)
        run("SFT", [py, "scripts/sft.py", "--config", str(auto_config("sft", choice))], env)

        stages = [("pretrain", ck / "pretrain" / "final.pt"), ("mid_m", ck / "mid_m" / "final.pt"),
                  ("sft", ck / "sft" / "final.pt"), ("mid_c", ck / "mid_c" / "final.pt")]
        for name, checkpoint in stages:
            output = RESULTS / f"{name}.json"
            if output.exists():
                log(f"=== avaliacao {name}: ja existe, pulando")
                continue
            if not checkpoint.exists():
                log(f"=== avaliacao {name}: {checkpoint} nao existe, pulando")
                continue
            run(f"avaliacao {name}", [py, "scripts/evaluate.py", "--checkpoint", str(checkpoint),
                                      "--output", str(output), "--val-dir", str(shared / "data/pretrain/val"),
                                      "--benchmarks", "hellaswag", "arc_easy",
                                      "--ppl-batch-size", "4", "--gsm-batch-size", "8"], env)

        for first, second in (("pretrain", "mid_m"), ("mid_c", "mid_m"), ("mid_m", "sft")):
            a, b = RESULTS / f"{first}.json", RESULTS / f"{second}.json"
            if a.exists() and b.exists():
                run(f"comparacao {first} x {second}", [py, "scripts/compare.py", str(a), str(b)], env)
        log("==================== tudo concluido ====================")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()

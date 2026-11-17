"""Motor de simulación de torneos de Axelrod (dilema del prisionero iterado).

Uso:
    uv run python -m tfm_axelrod.simulate --preset small --noise 0 0.05 \
        --turns 50 --repetitions 2 --seed 42

Modos:
- ``roundrobin``: todos contra todos (incluida la autopartida). Escribe una
  fila por jugada y por jugador en Parquet particionado por ``noise`` y
  ``turns`` (estilo Hive) bajo ``<out>/roundrobin``.
- ``evolutionary``: dinámica del replicador (``axelrod.Ecosystem``) sobre los
  resultados de un torneo round-robin, para cada nivel de ruido. Escribe en
  ``<out>/evolutionary``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import axelrod as axl
import pyarrow as pa
import pyarrow.parquet as pq

from tfm_axelrod.strategies import PRESETS, get_strategies

ROUNDROBIN_SCHEMA = pa.schema(
    [
        ("tournament_id", pa.string()),
        ("match_id", pa.int64()),
        ("repetition", pa.int32()),
        ("noise", pa.float64()),
        ("turns", pa.int32()),
        ("seed", pa.int64()),
        ("round", pa.int32()),
        ("player_index", pa.int8()),
        ("player", pa.string()),
        ("opponent", pa.string()),
        ("player_action", pa.string()),
        ("opponent_action", pa.string()),
        ("player_payoff", pa.int16()),
        ("opponent_payoff", pa.int16()),
        ("simulated_at", pa.timestamp("us", tz="UTC")),
    ]
)

EVOLUTIONARY_SCHEMA = pa.schema(
    [
        ("tournament_id", pa.string()),
        ("noise", pa.float64()),
        ("generation", pa.int32()),
        ("strategy", pa.string()),
        ("population_share", pa.float64()),
    ]
)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def make_tournament_id(mode: str, names: list[str], config: dict) -> str:
    """Identificador determinista: misma configuración -> mismo id."""
    payload = json.dumps({"mode": mode, "strategies": names, **config}, sort_keys=True)
    return f"{mode[:2]}_{hashlib.sha1(payload.encode()).hexdigest()[:12]}"


def match_seed(seed: int, noise: float, turns: int, repetition: int, p1: str, p2: str) -> int:
    """Semilla por partida, estable e independiente del orden/presets."""
    key = f"{seed}|{noise!r}|{turns}|{repetition}|{p1}|{p2}"
    return zlib.crc32(key.encode())


def _fmt(value: float) -> str:
    return repr(float(value))


# --------------------------------------------------------------------------- #
# Modo round-robin
# --------------------------------------------------------------------------- #
def run_roundrobin(
    strategies: list[type[axl.Player]],
    noise_levels: list[float],
    turns_list: list[int],
    repetitions: int,
    seed: int,
    out_dir: str | Path,
    batch_matches: int = 200,
) -> tuple[str, int]:
    """Juega todos contra todos y escribe Parquet por lotes.

    Devuelve ``(tournament_id, filas_escritas)``.
    """
    names = [cls().name for cls in strategies]
    tournament_id = make_tournament_id(
        "roundrobin",
        names,
        {"noise": [float(n) for n in noise_levels], "turns": list(turns_list),
         "repetitions": repetitions, "seed": seed},
    )
    simulated_at = datetime.now(timezone.utc)
    game = axl.Game()
    pairs = list(itertools.combinations_with_replacement(range(len(strategies)), 2))
    out_root = Path(out_dir) / "roundrobin"
    total_rows = 0
    match_id = 0

    for noise in noise_levels:
        for turns in turns_list:
            part_dir = out_root / f"noise={_fmt(noise)}" / f"turns={int(turns)}"
            part_dir.mkdir(parents=True, exist_ok=True)
            path = part_dir / f"{tournament_id}.parquet"
            cols: dict[str, list] = {f.name: [] for f in ROUNDROBIN_SCHEMA}
            pending = 0

            with pq.ParquetWriter(path, ROUNDROBIN_SCHEMA, compression="snappy") as writer:
                for rep in range(repetitions):
                    for i, j in pairs:
                        players = (strategies[i](), strategies[j]())
                        match = axl.Match(
                            players,
                            turns=turns,
                            noise=noise,
                            game=game,
                            seed=match_seed(seed, noise, turns, rep, names[i], names[j]),
                        )
                        interactions = match.play()
                        scores = match.scores()
                        _append_match(
                            cols, interactions, scores, (names[i], names[j]),
                            tournament_id, match_id, rep, noise, turns, seed, simulated_at,
                        )
                        match_id += 1
                        pending += 1
                        if pending >= batch_matches:
                            total_rows += _flush(writer, cols)
                            pending = 0
                if pending:
                    total_rows += _flush(writer, cols)

    return tournament_id, total_rows


def _append_match(cols, interactions, scores, names, tournament_id, match_id, rep,
                  noise, turns, seed, simulated_at) -> None:
    n = len(interactions)
    for idx in (0, 1):
        opp = 1 - idx
        cols["tournament_id"].extend([tournament_id] * n)
        cols["match_id"].extend([match_id] * n)
        cols["repetition"].extend([rep] * n)
        cols["noise"].extend([float(noise)] * n)
        cols["turns"].extend([int(turns)] * n)
        cols["seed"].extend([seed] * n)
        cols["round"].extend(range(1, n + 1))
        cols["player_index"].extend([idx] * n)
        cols["player"].extend([names[idx]] * n)
        cols["opponent"].extend([names[opp]] * n)
        cols["player_action"].extend(str(a[idx]) for a in interactions)
        cols["opponent_action"].extend(str(a[opp]) for a in interactions)
        cols["player_payoff"].extend(int(s[idx]) for s in scores)
        cols["opponent_payoff"].extend(int(s[opp]) for s in scores)
        cols["simulated_at"].extend([simulated_at] * n)


def _flush(writer: pq.ParquetWriter, cols: dict[str, list]) -> int:
    batch = pa.RecordBatch.from_pydict(cols, schema=ROUNDROBIN_SCHEMA)
    writer.write_batch(batch)
    for v in cols.values():
        v.clear()
    return batch.num_rows


# --------------------------------------------------------------------------- #
# Modo evolutivo
# --------------------------------------------------------------------------- #
def run_evolutionary(
    strategies: list[type[axl.Player]],
    noise_levels: list[float],
    turns: int,
    repetitions: int,
    seed: int,
    out_dir: str | Path,
    generations: int = 100,
) -> tuple[str, int]:
    """Dinámica del replicador entre las estrategias, por nivel de ruido.

    Para cada ruido se juega un torneo round-robin (``axelrod.Tournament``)
    y sobre su matriz de pagos se itera ``axelrod.Ecosystem`` partiendo de
    una población uniforme. Devuelve ``(tournament_id, filas_escritas)``.
    """
    names = [cls().name for cls in strategies]
    tournament_id = make_tournament_id(
        "evolutionary",
        names,
        {"noise": [float(n) for n in noise_levels], "turns": int(turns),
         "repetitions": repetitions, "seed": seed, "generations": generations},
    )
    out_root = Path(out_dir) / "evolutionary"
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{tournament_id}.parquet"
    total_rows = 0

    with pq.ParquetWriter(path, EVOLUTIONARY_SCHEMA, compression="snappy") as writer:
        for noise in noise_levels:
            players = [cls() for cls in strategies]
            tournament = axl.Tournament(
                players, turns=turns, noise=noise, repetitions=repetitions,
                seed=match_seed(seed, noise, turns, 0, "tournament", "evolutionary"),
            )
            results = tournament.play(progress_bar=False, processes=1)
            eco = axl.Ecosystem(results)
            eco.reproduce(generations)
            rows = {f.name: [] for f in EVOLUTIONARY_SCHEMA}
            for gen, shares in enumerate(eco.population_sizes):
                for name, share in zip(names, shares):
                    rows["tournament_id"].append(tournament_id)
                    rows["noise"].append(float(noise))
                    rows["generation"].append(gen)
                    rows["strategy"].append(name)
                    rows["population_share"].append(float(share))
            batch = pa.RecordBatch.from_pydict(rows, schema=EVOLUTIONARY_SCHEMA)
            writer.write_batch(batch)
            total_rows += batch.num_rows

    return tournament_id, total_rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tfm_axelrod.simulate",
        description="Simulación de torneos de Axelrod (dilema del prisionero iterado).",
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--noise", type=float, nargs="+", default=[0.0])
    p.add_argument("--turns", type=int, nargs="+", default=[200])
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--mode", choices=["roundrobin", "evolutionary"], default="roundrobin")
    p.add_argument("--generations", type=int, default=100,
                   help="Generaciones de la dinámica evolutiva (solo modo evolutionary)")
    p.add_argument("--batch-matches", type=int, default=200,
                   help="Partidas acumuladas antes de escribir un lote (roundrobin)")
    p.add_argument("--upload-prefix", default=None,
                   help="Si se indica, al terminar sube los Parquet generados al data lake "
                        "bajo este prefijo, manteniendo la estructura de carpetas de --out "
                        "(requiere AZURE_STORAGE_CONNECTION_STRING y el extra 'azure')")
    p.add_argument("--upload-filesystem", default="lake",
                   help="Sistema de ficheros ADLS Gen2 de destino (por defecto: lake)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for n in args.noise:
        if not 0.0 <= n <= 1.0:
            raise SystemExit(f"--noise fuera de [0, 1]: {n}")
    strategies = get_strategies(args.preset)

    fs_client = None
    if args.upload_prefix is not None:
        from tfm_axelrod.cloud import UploadError, check_filesystem, get_filesystem_client

        # Falla antes de simular si falta configuración o no hay acceso al lake.
        try:
            fs_client = get_filesystem_client(args.upload_filesystem)
            check_filesystem(fs_client)
        except UploadError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc

    if args.mode == "roundrobin":
        tid, rows = run_roundrobin(
            strategies, args.noise, args.turns, args.repetitions, args.seed,
            args.out, batch_matches=args.batch_matches,
        )
        where = Path(args.out) / "roundrobin"
    else:
        if len(args.turns) > 1:
            print(f"[aviso] modo evolutionary usa solo --turns {args.turns[0]}")
        tid, rows = run_evolutionary(
            strategies, args.noise, args.turns[0], args.repetitions, args.seed,
            args.out, generations=args.generations,
        )
        where = Path(args.out) / "evolutionary"

    print(f"tournament_id={tid} filas={rows} salida={where}", flush=True)

    if fs_client is not None:
        from tfm_axelrod.cloud import UploadError, upload_files

        files = sorted(where.rglob(f"{tid}.parquet"))
        expected = len(args.noise) * len(args.turns) if args.mode == "roundrobin" else 1
        if len(files) != expected:
            raise SystemExit(f"ERROR: se esperaban {expected} ficheros Parquet y hay {len(files)}")
        print(f"Subiendo {len(files)} ficheros a {args.upload_filesystem}/"
              f"{args.upload_prefix.strip('/')}/ ...", flush=True)
        try:
            uploaded = upload_files(files, args.out, args.upload_prefix, fs_client)
        except UploadError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        print(f"RESUMEN filas={rows} ficheros_subidos={len(uploaded)}", flush=True)


if __name__ == "__main__":
    main()

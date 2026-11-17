"""Compacta la capa raw local en un Parquet por partición para subirla al data lake."""
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

RAW = Path("data/raw/roundrobin")
OUT = Path("data/upload")
OUT.mkdir(parents=True, exist_ok=True)

total = 0
for part in sorted(p for p in RAW.glob("noise=*/turns=*") if p.is_dir()):
    table = ds.dataset(part).to_table()
    name = f"roundrobin_{part.parent.name}_{part.name}.parquet".replace("=", "")
    pq.write_table(table, OUT / name, compression="zstd")
    total += table.num_rows
    print(name, table.num_rows)

print("Total filas:", total)
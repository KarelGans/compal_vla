"""Copy the metrics from a lerobot-train log file into the Trackio dashboard.

Use it for runs that were trained before Trackio was switched on, so their curves show up next
to newer runs. The run is named after the log file, the same name train.sh and resume.sh use, so
resuming that run continues the same curves. Steps that are already in the dashboard are skipped,
so running it twice is safe.

    source env.sh
    uv run python scripts/log_to_trackio.py pi05_open_box_20261114_1239            # pi05 run name, or a .log path
    uv run python scripts/log_to_trackio.py --model act <run>                       # an ACT run
    uv run trackio show --project pi05_open_box
"""
import argparse
import os
import re
from pathlib import Path

import trackio
from lerobot.utils.utils import format_big_number
from trackio.sqlite_storage import SQLiteStorage

# log abbreviation -> name lerobot-train uses when it logs to Trackio itself (prefixed with "train/")
METRICS = {
    "epch": "epochs",
    "loss": "loss",
    "grdn": "grad_norm",
    "lr": "lr",
    "data_s": "dataloading_s",
    "prep_s": "preprocessing_s",
    "updt_s": "update_s",
    "step_s": "step_s",
    "smp/s": "samples_per_s",
    "mem_gb": "gpu_mem_gb",
}
# The INFO line lerobot-train prints every log_freq steps. It rounds the step ("step:12K"), so the
# exact step is taken from the progress bar ("12300/50000 [...]") written just before it.
LINE = re.compile(r"\bstep:(\S+) (.*)$")
PROGRESS = re.compile(r"(\d+)/\d+ \[")


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        bar = PROGRESS.findall(line[: m.start()])
        step = int(bar[-1]) if bar else int(m.group(1)) if m.group(1).isdigit() else None
        if step is None or format_big_number(step) != m.group(1):
            continue  # no exact step for this line
        values = {"train/steps": step}
        for token in m.group(2).split():
            key, _, value = token.partition(":")
            if key in METRICS:
                values[f"train/{METRICS[key]}"] = float(value)
        if "train/loss" in values:
            rows[step] = values  # a crash + resume repeats steps in the log: keep the later run's values
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run name (log in $CHECKPOINT_ROOT/<model>/logs/) or path to a .log file")
    p.add_argument("--model", default="pi05", help="model folder: pi05, act")
    p.add_argument("--project", default=None, help="default: <model>_open_box, as train.sh uses")
    args = p.parse_args()
    args.project = args.project or f"{args.model}_open_box"

    log = Path(args.run)
    if not log.is_file():
        log = Path(os.environ["CHECKPOINT_ROOT"]) / args.model / "logs" / f"{args.run}.log"
    run = log.stem
    rows = parse_log(log)
    if not rows:
        raise SystemExit(f"No training metrics found in {log}")

    done = {r["step"] for r in SQLiteStorage.get_logs(args.project, run, scalar_only=True) if "train/loss" in r}
    todo = sorted(set(rows) - done)
    print(f"{log}: {len(rows)} logged steps ({min(rows)}-{max(rows)}), {len(done)} already in the dashboard")
    if not todo:
        return
    trackio.init(project=args.project, name=run, resume="allow",
                 auto_log_gpu=False, auto_log_cpu=False, capture_console=False)
    for step in todo:
        trackio.log(rows[step], step=step)
    trackio.finish()
    print(f"added {len(todo)} steps to project '{args.project}', run '{run}'  (data in {os.environ.get('TRACKIO_DIR')})")


if __name__ == "__main__":
    main()

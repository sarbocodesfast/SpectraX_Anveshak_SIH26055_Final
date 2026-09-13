"""Train any tier on the full published corpus. Paste into a Deepnote cell.

Set the tier when you run it::

    SMARTSCAN_TIER=medium python /tmp/c.py
    SMARTSCAN_TIER=easy   python /tmp/c.py

This is the HARD cell generalised, carrying everything that run had to learn:

* the corpus is unpacked to LOCAL disk. ``/work`` is an s3fs mount and
  unpacking the 9003-file archive there was still going after 15 minutes
  against 10 seconds locally;
* ``run.out_dir`` points at ``/work``, so the trainer's checkpoints land on
  persistent storage as it goes. The first HARD run reached student epoch 9/12
  with a best validation AP of 0.8086 and then the machine stopped and the
  model was gone, because checkpoints were on ``/root`` and the copy to
  ``/work`` was the last line of the cell;
* batch 64 hit CUDA OOM on the L4 (9.58 GiB wanted against 8.57 GiB reserved
  but unallocated, i.e. fragmentation), so expandable segments and a ladder
  down through 32/16/8 rather than one guess;
* the corpus is verified before the hours are spent -- the run refuses to
  continue if the loader silently falls back to regenerating from seeds;
* no Kaggle credentials, because the archive is already on ``/work``.

**The window budget is held constant across tiers.** MEDIUM has about half
again as many episodes as HARD, so a fixed windows-per-episode would make it
half again as long for no extra episode coverage. Instead the per-episode
count is derived from a fixed total, so every tier trains on *all* of its
episodes for roughly the same wall clock, and the comparison between tiers is
not confounded by how long each one was allowed to run.

Epoch counts come from what the HARD run showed: its teacher flattened at
0.0313 by epoch 3 and sat at 0.0312 through epoch 10, and its student went
0.7978 at epoch 1 to 0.8086 at epoch 7 -- six epochs for 1.4%. So 3 and 6.
"""
import json
import os
import pathlib
import subprocess
import time
import zipfile

T0 = time.time()
TIER = os.environ.get("SMARTSCAN_TIER", "medium")
ARCHIVE = "/work/kagglehub_cache/datasets/shirishrajgupta/ew-smart-scan-rf-environment/3.archive"
DS = "/root/ds"
REPO = "/root/repo"
OUT = f"/work/trainout_{TIER}"

#: Total training windows, held constant across tiers so wall clock is
#: comparable and no tier is advantaged by simply being allowed to run longer.
#: Matches what the HARD run used (557 episodes x 120).
TARGET_WINDOWS = 67_000


def sh(cmd, check=False):
    """Run a shell command, streaming its output into the cell."""
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"  -> exit {rc}", flush=True)
    if check and rc:
        raise SystemExit(f"failed: {cmd}")
    return rc


print("=" * 62 + f"\nTIER = {TIER}\n" + "=" * 62, flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh(f"ls -la {OUT}/checkpoints/ 2>/dev/null || echo '(no previous checkpoints)'")

if not pathlib.Path(ARCHIVE).exists():
    raise SystemExit(
        f"The corpus archive is gone from {ARCHIVE}.\n"
        "Re-download needs KAGGLE_USERNAME and KAGGLE_KEY as Deepnote environment\n"
        "variables, then: pip install kagglehub && python -c \"import kagglehub;"
        "kagglehub.dataset_download('shirishrajgupta/ew-smart-scan-rf-environment')\""
    )

# ---- corpus onto local disk -------------------------------------------
if not list(pathlib.Path(DS).rglob("index.parquet")):
    sh(f"rm -rf {DS} && mkdir -p {DS}")
    print(f"extracting {ARCHIVE} -> {DS}", flush=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(DS)
found = list(pathlib.Path(DS).rglob("index.parquet"))
if not found:
    raise SystemExit("no index.parquet -- the corpus did not unpack")
DS_ROOT = str(found[0].parent)
print("corpus root:", DS_ROOT, flush=True)

# ---- code and deps ----------------------------------------------------
if not pathlib.Path(REPO, ".git").exists():
    sh(f"rm -rf {REPO} && git clone --depth 1 "
       f"https://github.com/shirish-raj-gupta/SIH26055_Prototype.git {REPO}", check=True)
else:
    sh(f"cd {REPO} && git fetch -q --all && git reset -q --hard origin/main")
sh("python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121")
sh(f"cd {REPO} && python -m pip install -q -e '.[ml,viz]'", check=True)

# Seed the persistent output dir with the shipped model so the NEW/SHIP
# comparison has something to compare against on a fresh machine.
sh(f"mkdir -p {OUT}/checkpoints")
for f in (f"predictor_{TIER}.pt", f"predictor_{TIER}_history.json"):
    src = pathlib.Path(REPO) / "runs" / "checkpoints" / f
    dst = pathlib.Path(OUT) / "checkpoints" / f.replace(".", "_shipped.", 1)
    if src.exists() and not dst.exists():
        dst.write_bytes(src.read_bytes())
        print(f"kept shipped {f} -> {dst.name}", flush=True)

# ---- prove the corpus is real before spending hours -------------------
if sh(f"cd {REPO} && python scripts/deepnote_hard_train.py --stage data "
      f"--tier {TIER} --dataset-root {DS_ROOT}"):
    raise SystemExit("corpus check failed -- refusing to train on regenerated data")

# ---- how many windows per episode, for a constant total ---------------
import sys  # noqa: E402  (only needed once the repo is on the path)

sys.path.insert(0, REPO)
from smartscan.data.kaggle_io import load_dataset  # noqa: E402

n_train = len(load_dataset("train", tier=TIER, root=DS_ROOT, allow_download=False))
wpe = max(20, TARGET_WINDOWS // max(n_train, 1))
print(f"\n{n_train} train episodes -> {wpe} windows each "
      f"(~{n_train * wpe:,} total, target {TARGET_WINDOWS:,})", flush=True)

# ---- train, checkpointing straight to /work ---------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
trained = False
for bs in (32, 16, 8):
    print(f"\n{'=' * 62}\n{TIER} predictor, batch_size={bs}\n{'=' * 62}", flush=True)
    rc = sh(f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"python -m smartscan.cli train --config {TIER}.yaml --what predictor "
            f"--dataset {DS_ROOT} --windows-per-episode {wpe} --workers 16 --steps 6 "
            f"--set predictor.batch_size={bs} "
            f"--set predictor.distillation.teacher_epochs=3 "
            f"--set run.out_dir={OUT}")
    if rc == 0:
        trained = True
        break
    print(f"batch_size={bs} failed (likely OOM); trying smaller", flush=True)

print(f"\ntrained={trained} after {(time.time() - T0) / 60:.1f} min", flush=True)
sh(f"ls -la {OUT}/checkpoints/")


# ---- did it get better? -----------------------------------------------
def scores(path):
    """(auc, ap, accuracy, lift, base rate, best val AP) or None."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    s = h.get("scores_vs_truth") or {}
    return (s.get("auc"), s.get("average_precision"), s.get("accuracy"),
            h.get("ap_lift_over_base_rate"), s.get("positive_rate"),
            h.get("best_val_ap"))


print("\n" + "=" * 62 + "\nPREDICTOR QUALITY\n" + "=" * 62, flush=True)
for tag, fname in (("NEW ", f"predictor_{TIER}_history.json"),
                   ("SHIP", f"predictor_{TIER}_history_shipped.json")):
    sc = scores(pathlib.Path(OUT) / "checkpoints" / fname)
    if sc is None:
        print(f"{tag} (missing {fname})", flush=True)
    else:
        auc, ap, acc, lift, base, bva = (x if x is not None else float("nan") for x in sc)
        print(f"{tag} auc={auc:.4f} ap={ap:.4f} acc={acc:.4f} lift={lift:.2f}x "
              f"base={base:.4f} best_val_ap={bva:.4f}", flush=True)

# ---- the number that decides it ---------------------------------------
# Not the loss and not the AP. On MEDIUM, retraining once raised AUC
# 0.684 -> 0.763 and made the scheduler WORSE: never-intercepted 112 -> 126.
sh(f"cp -f {OUT}/checkpoints/predictor_{TIER}.pt {REPO}/runs/checkpoints/ 2>/dev/null")
sh(f"cd {REPO} && python scripts/deepnote_hard_train.py --stage evaluate "
   f"--tier {TIER} --n-seeds 8")
sh(f"cp -v {REPO}/reports/hard_retrain_eval.json {OUT}/ 2>/dev/null")
sh(f"ls -la {OUT}/")
print(f"\n=== DONE ({TIER}) in {(time.time() - T0) / 60:.1f} min ===", flush=True)
print(f"Everything worth keeping is under {OUT} on persistent storage.", flush=True)

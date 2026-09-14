"""Architecture control on Kaggle: transformer vs GRU, everything else frozen.

Paste into one Kaggle notebook cell. Requires **Internet: On** (to clone and
pip install) and an accelerator; T4 x2 or P100 both work.

Only the architecture varies. Both arms take the same ``--episodes`` path, so
window construction, the 80/20 split and the held-out set are identical
between them, and the two average precisions are measurements of one
quantity. That was not true of the corpus comparison this replaces: its two
arms validated on different held-out sets, so their APs were never
comparable.

**On T4 x2.** The trainer selects a single device -- there is no
``DataParallel`` here -- so the second GPU is idle. That costs nothing for
this job: 200 episodes of windows are held in memory and one T4 is ample. Do
not read the second GPU's idleness as a fault.

**Why this is fast on Kaggle when a previous Kaggle run was not.** The
earlier attempt used ``--dataset``, which re-reads every window from parquet
on every epoch; two T4 runs spent about 14 hours reaching teacher epoch 1 of
4 at ~9.5 s/batch, against 27 ms/step for the same model on an in-memory
batch. This script never touches that path.

Results land in ``/kaggle/working/arch_control_<tier>/summary.json``, which
Kaggle keeps as notebook output.
"""

import json
import os
import pathlib
import subprocess
import time

T0 = time.time()

ARCHS = ("transformer", "gru")          # the only thing that varies

# ---- frozen ------------------------------------------------------------
TIER = os.environ.get("SMARTSCAN_TIER", "medium")
BATCH = 32          # pinned, NOT a ladder: batch 8 vs 64 moved AP 0.6915->0.7457
EPOCHS = 6
TEACHER_EPOCHS = 3
EPISODES = 200
WPE = 200
SEED = 0            # same episodes for both arms
WORKERS = 4         # Kaggle gives 4 vCPU; 8 would oversubscribe

REPO = "/kaggle/working/ctlrepo"
OUT = "/kaggle/working/arch_control_" + TIER
REPO_URL = "https://github.com/shirish-raj-gupta/SIH26055_Prototype"


def sh(cmd, check=True):
    """Run a shell command, streaming its output into the cell."""
    print("\n$ " + cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print("  -> exit " + str(rc), flush=True)
    if rc and check:
        raise SystemExit("failed: " + cmd)
    return rc


# ---- environment -------------------------------------------------------
# Kaggle images already ship a CUDA torch. Reinstalling it would burn several
# minutes and, worse, could change the torch build mid-experiment -- a rerun
# of this control elsewhere silently moved from 2.5.1+cu121 to 2.14.0+cu130
# because a pip index resolved differently. Both arms must share one build.
import torch  # noqa: E402

print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"n_gpu={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)}", flush=True)

sh(f"rm -rf {REPO} && git clone -q {REPO_URL} {REPO}")
sh(f"git -C {REPO} log --oneline -1")
# No [ml] extra: that would pull a different torch over Kaggle's.
sh(f'python -m pip install -q -e "{REPO}" 2>&1 | tail -2')
sh(f"mkdir -p {OUT}")

# ---- the two arms ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
results = {}
for arch in ARCHS:
    name = f"{TIER}-{arch}"
    print("\n" + "=" * 64)
    print(f"{name}   batch={BATCH} episodes={EPISODES} wpe={WPE} seed={SEED}")
    print("=" * 64, flush=True)
    t = time.time()
    rc = sh(
        f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"python -m smartscan.cli train --config {TIER}.yaml --what predictor "
        f"--arch {arch} --episodes {EPISODES} --windows-per-episode {WPE} "
        f"--workers {WORKERS} --steps {EPOCHS} "
        f"--set run.seed={SEED} "
        f"--set predictor.batch_size={BATCH} "
        f"--set predictor.distillation.teacher_epochs={TEACHER_EPOCHS} "
        f"--set run.out_dir={OUT}/{name}",
        check=False)
    results[name] = {"rc": rc, "minutes": round((time.time() - t) / 60, 1)}
    print(f"  {name}: exit {rc} in {results[name]['minutes']} min", flush=True)

# ---- result ------------------------------------------------------------
print("\n" + "=" * 64)
print("RESULT  (student = the deployed, observation-only model)")
print("=" * 64, flush=True)
table = {}
for name, meta in results.items():
    hist = pathlib.Path(OUT) / name / "checkpoints" / f"predictor_{TIER}_history.json"
    if not hist.is_file():
        print(f"{name:26} FAILED (no history) {meta['minutes']} min")
        continue
    h = json.loads(hist.read_text())
    # `scores_vs_truth` is the STUDENT. `teacher_scores_vs_truth` sees
    # privileged state and is not a deployable candidate.
    s = h.get("scores_vs_truth") or {}
    table[name] = {"ap": s.get("average_precision"), "auc": s.get("auc"),
                   "base": s.get("positive_rate"), "arch_recorded": h.get("arch"),
                   "torch": torch.__version__}
    table[name].update(meta)
    print(f"{name:26} AP={table[name]['ap']:.4f} AUC={table[name]['auc']:.4f} "
          f"base={table[name]['base']:.4f} ({meta['minutes']} min)")

(pathlib.Path(OUT) / "summary.json").write_text(json.dumps(table, indent=2))

t_ap = (table.get(f"{TIER}-transformer") or {}).get("ap")
g_ap = (table.get(f"{TIER}-gru") or {}).get("ap")
print("\n" + "-" * 64)
if t_ap and g_ap:
    print(f"ARCHITECTURE EFFECT (gru vs transformer): "
          f"{(g_ap - t_ap) / abs(t_ap) * 100:+.1f}% AP")
    print("Same episodes, same windows, same held-out split, same recipe,")
    print("same torch build. Architecture is the only thing that differs.")
else:
    print("Incomplete: at least one arm produced no model.")
print(f"\nwrote {OUT}/summary.json")
print(f"total {(time.time() - T0) / 60:.1f} min", flush=True)
print("ALLDONE_ARCHCTL")

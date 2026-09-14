"""Architecture control on Kaggle: transformer vs GRU, everything else frozen.

Paste into one Kaggle notebook cell. Requires **Internet: On** (to clone and
pip install) and the accelerator set to **T4 x2**.

**Not P100.** A P100 is compute capability sm_60 and Kaggle's preinstalled
torch supports sm_70 and above, so training falls back to CPU: the
transformer arm took 64 minutes instead of 4, and the GRU arm projected to
16.5 h against a 12 h limit. The script now launches a CUDA kernel up front
and refuses to start rather than discovering this an hour in.

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
import sys
import time

T0 = time.time()

ARCHS = ("transformer", "gru")          # the only thing that varies

# ---- frozen ------------------------------------------------------------
TIER = os.environ.get("SMARTSCAN_TIER", "medium")
BATCH = 32          # pinned, NOT a ladder: batch 8 vs 64 moved AP 0.6915->0.7457
EPOCHS = 3          # transformer's best epoch was 2 of 6; the rest overfit
TEACHER_EPOCHS = 2
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

# Make the GPU usable, or stop -- but do not train on CPU by accident.
#
# Kaggle keeps assigning a Tesla P100 regardless of the accelerator requested
# through the API, and a P100 is compute capability sm_60 while the
# preinstalled torch is built for sm_70 and above. `_pick_device` correctly
# probes with a real kernel, sees it fail, and falls back to CPU -- which took
# 64 minutes for an arm that needs 4, with the GRU arm heading for 16.5 h
# against a 12 h limit.
#
# The fix is a torch build that includes sm_60. cu121 ships sm_50 through
# sm_90, so it covers the P100 and the T4 both. Installing it costs a few
# minutes and is recorded in the manifest; crucially BOTH arms then share one
# build, so the comparison is unaffected.
def _gpu_runs_kernels() -> tuple[bool, str]:
    """Probe the GPU in a subprocess, so a bad build cannot poison this one."""
    probe = (
        "import torch;"
        "torch.zeros(8,8,device='cuda') @ torch.zeros(8,8,device='cuda');"
        "print('OK', torch.__version__, torch.cuda.get_device_name(0))"
    )
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout or r.stderr).strip().splitlines()[-1][:160]


ok, detail = _gpu_runs_kernels()
print(f"GPU probe: {'OK' if ok else 'FAILED'} -- {detail}", flush=True)
if not ok:
    print()
    print("Installing a torch build that supports this GPU (cu121 covers "
          "sm_50-sm_90)...", flush=True)
    sh("python -m pip install -q torch --index-url "
       "https://download.pytorch.org/whl/cu121 2>&1 | tail -2", check=False)
    ok, detail = _gpu_runs_kernels()
    print(f"GPU probe after reinstall: {'OK' if ok else 'FAILED'} -- {detail}",
          flush=True)
if not ok:
    raise SystemExit(
        "still cannot launch a CUDA kernel: " + detail + "\n"
        "Training would fall back to CPU and take ~16x longer, so stopping "
        "here rather than burning hours. Set the accelerator to T4 x2."
    )

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

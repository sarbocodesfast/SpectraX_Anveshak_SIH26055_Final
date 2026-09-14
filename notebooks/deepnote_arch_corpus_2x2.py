"""Corpus x architecture control, 2x2. Paste into one Deepnote cell and run.

Every "full corpus improved prediction" number in this project is quarantined,
because the run that produced them moved five things at once: it passed no
``--arch`` (so it silently took the ``gru`` default while the shipped weights
were transformers), tried ``batch_size`` in a ``(32, 16, 8)`` ladder without
recording which rung succeeded, and overrode ``--steps`` and
``teacher_epochs`` besides. Five factors moved, so the effect belongs to all
five jointly and to none of them separately.

This cell moves exactly two, and freezes everything else::

                     | transformer | gru
    -----------------|-------------|-----
    seeds corpus     |     arm     | arm
    published corpus |     arm     | arm

**What is frozen, and why each one had to be.** Batch size is pinned rather
than laddered, because retraining one arm at 8 instead of 64 moved student AP
0.6915 -> 0.7457 -- larger than several effects that had been credited to the
corpus. Episode count and windows-per-episode are pinned across *both*
corpora, so "corpus" means which episodes rather than how many; the first
attempt at this control read 869 published episodes against 200 regenerated
ones and spent 28 minutes data-bound with the GPU at 0%. Epochs, teacher
epochs, tier and seed are pinned for the ordinary reason.

Checkpoints are written under ``/work`` from the start. An earlier run kept
them on ephemeral ``/root`` and copied at the end; the machine stopped at
epoch 9 of 12 and the whole run was lost.

Each arm writes a full training manifest, so ``assert_comparable`` can license
the decomposition afterwards instead of us asserting it.
"""

import json
import os
import pathlib
import subprocess
import time
import zipfile

T0 = time.time()

# ---- the two factors that move -----------------------------------------
ARCHS = ("transformer", "gru")
CORPORA = ("seeds", "dataset")

# ---- everything else, frozen -------------------------------------------
TIER = os.environ.get("SMARTSCAN_TIER", "medium")
BATCH = 32          # pinned, NOT a ladder
EPOCHS = 6
TEACHER_EPOCHS = 3
EPISODES = 200      # identical for both corpora
WPE = 200           # windows per episode, identical for both corpora
WORKERS = 8

REPO = "/root/ctlrepo"
DS = "/root/ctlds"                       # local disk: /work is s3fs and slow
OUT = "/work/arch_corpus_2x2_" + TIER
ARCHIVE = ("/work/kagglehub_cache/datasets/shirishrajgupta/"
           "ew-smart-scan-rf-environment/3.archive")
REPO_URL = "https://github.com/shirish-raj-gupta/SIH26055_Prototype"


def sh(cmd, check=True):
    """Run a shell command, streaming its output into the cell."""
    print("\n$ " + cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print("  -> exit " + str(rc), flush=True)
    if rc and check:
        raise SystemExit("failed: " + cmd)
    return rc


# ---- refuse to run twice at once ---------------------------------------
# A previous attempt died on FileNotFoundError from os.getcwd(). The cause was
# not the machine: this script had been pasted into the notebook as a block,
# and Deepnote auto-runs the notebook when a session is created, so every
# session spawned another instance. Each instance begins by deleting and
# re-cloning REPO -- which is the running instance's working directory. They
# destroyed each other.
#
# The lock is the fix, and it is kept even though the block has been removed,
# because "only one copy of this is running" is a property the experiment
# needs rather than a habit of how it happens to be launched.
LOCK = pathlib.Path("/work/_jobs/arch2x2.lock")
LOCK.parent.mkdir(parents=True, exist_ok=True)
if LOCK.exists():
    age_min = (time.time() - LOCK.stat().st_mtime) / 60
    if age_min < 240:
        raise SystemExit(
            f"another instance started {age_min:.0f} min ago ({LOCK}). "
            "Two instances delete each other's working directory. Remove the "
            "lock only once you have confirmed nothing is running.")
    print(f"stale lock ({age_min:.0f} min old), taking it", flush=True)
LOCK.write_text(str(os.getpid()))

# ---- environment -------------------------------------------------------
# A stopped machine wipes /root, venv included, so never assume torch is here.
sh(f"rm -rf {REPO} && git clone -q {REPO_URL} {REPO}")
sh(f"git -C {REPO} log --oneline -1")
sh("python -m pip install -q torch --index-url "
   "https://download.pytorch.org/whl/cu121")
sh(f'python -m pip install -q -e "{REPO}[ml,viz]"')

import torch  # noqa: E402  (only meaningful once the install above has run)

_dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
print(f"\ntorch {torch.__version__} cuda={torch.cuda.is_available()} {_dev}", flush=True)

# ---- corpus onto local disk -------------------------------------------
if not pathlib.Path(ARCHIVE).exists():
    raise SystemExit(
        "The corpus archive is gone from " + ARCHIVE + ".\n"
        "Re-download it with your own Kaggle credentials:\n"
        "  pip install kagglehub && python -c \"import kagglehub; "
        "kagglehub.dataset_download('shirishrajgupta/"
        "ew-smart-scan-rf-environment')\"")


def ensure_corpus(force=False):
    """Unpack the corpus to local disk, and return its root.

    Re-checked before every arm rather than once at the start. ``/root`` is
    ephemeral: a first attempt at this control unpacked the corpus, trained
    one arm for four minutes, and then died on ``FileNotFoundError`` because
    the machine had been recycled underneath it and taken ``/root/ctlds``
    with it. Re-unpacking costs a couple of minutes; losing the run costs the
    whole experiment.
    """
    marker = list(pathlib.Path(DS).rglob("index.parquet")) if not force else []
    if marker:
        return marker[0].parent
    sh(f"rm -rf {DS} && mkdir -p {DS} {OUT}")
    print("\nunpacking corpus to local disk...", flush=True)
    zipfile.ZipFile(ARCHIVE).extractall(DS)
    found = list(pathlib.Path(DS).rglob("index.parquet"))
    if not found:
        raise SystemExit("no index.parquet -- the corpus did not unpack")
    print("corpus root: " + str(found[0].parent), flush=True)
    return found[0].parent


sh(f"mkdir -p {DS} {OUT}")
DS_ROOT = ensure_corpus()

# ---- the four arms -----------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
results = {}
for arch in ARCHS:
    for corpus in CORPORA:
        name = f"{TIER}-{arch}-{corpus}"
        print("\n" + "=" * 64)
        print(f"{name}   batch={BATCH} episodes={EPISODES} wpe={WPE}")
        print("=" * 64, flush=True)
        if corpus == "dataset":
            DS_ROOT = ensure_corpus()          # /root may have been recycled
            src = f"--dataset {DS_ROOT} --dataset-episodes {EPISODES} --windows-per-episode {WPE}"
        else:
            src = f"--episodes {EPISODES} --windows-per-episode {WPE}"
        # The repo lives on ephemeral disk too, and a missing one fails every
        # remaining arm rather than just this one.
        if not pathlib.Path(REPO).is_dir():
            sh(f"git clone -q {REPO_URL} {REPO}")
            sh(f'python -m pip install -q -e "{REPO}[ml,viz]"')
        t = time.time()
        rc = sh(
            f"cd {REPO} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"python -m smartscan.cli train --config {TIER}.yaml "
            f"--what predictor --arch {arch} {src} --workers {WORKERS} "
            f"--steps {EPOCHS} "
            f"--set predictor.batch_size={BATCH} "
            f"--set predictor.distillation.teacher_epochs={TEACHER_EPOCHS} "
            f"--set run.out_dir={OUT}/{name}",
            check=False)
        results[name] = {"rc": rc, "minutes": round((time.time() - t) / 60, 1)}
        print("  {}: exit {} in {} min".format(
            name, rc, results[name]["minutes"]), flush=True)

# ---- the decomposition -------------------------------------------------
print("\n" + "=" * 64)
print("RESULTS  (student = the deployed, observation-only model)")
print("=" * 64, flush=True)
print("{:34} {:>8} {:>8} {:>7} {:>6}".format(
    "arm", "AP", "AUC", "base", "min"))

def fmt(v):
    """Format a score, or a dash when the arm did not produce one."""
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


table = {}
for name, meta in results.items():
    hist = (pathlib.Path(OUT) / name / "checkpoints" /
            ("predictor_" + TIER + "_history.json"))
    if not hist.is_file():
        print("{:34} {:>30} {:>6}".format(
            name, "FAILED (no history)", meta["minutes"]))
        continue
    h = json.loads(hist.read_text())
    s = h.get("student") or h.get("student_scores") or {}
    ap, auc = s.get("average_precision"), s.get("auc")
    base = h.get("positive_base_rate") or h.get("base_rate")
    table[name] = {"ap": ap, "auc": auc, "base": base}
    table[name].update(meta)
    print("{:34} {:>8} {:>8} {:>7} {:>6}".format(
        name, fmt(ap), fmt(auc), fmt(base), meta["minutes"]))

(pathlib.Path(OUT) / "summary.json").write_text(json.dumps(table, indent=2))


def gap(a, b, field="ap"):
    """B minus a, as a percentage of a."""
    x = table.get(a, {}).get(field)
    y = table.get(b, {}).get(field)
    if not x or not y:
        return None
    return round((y - x) / abs(x) * 100, 1)


print("\n" + "-" * 64)
print("MAIN EFFECTS (student AP)")
print("-" * 64)
print("  corpus effect, transformer : {} %".format(
    gap(TIER + "-transformer-seeds", TIER + "-transformer-dataset")))
print("  corpus effect, gru         : {} %".format(
    gap(TIER + "-gru-seeds", TIER + "-gru-dataset")))
print("  arch effect, seeds corpus  : {} %".format(
    gap(TIER + "-transformer-seeds", TIER + "-gru-seeds")))
print("  arch effect, published     : {} %".format(
    gap(TIER + "-transformer-dataset", TIER + "-gru-dataset")))
print("\nIf the two corpus effects disagree in sign or size, corpus and "
      "architecture interact and neither has a single main effect to quote.")
print("\nwrote " + OUT + "/summary.json")
print(f"total {(time.time() - T0) / 60:.1f} min", flush=True)
LOCK.unlink(missing_ok=True)
print("ALLDONE_2X2")

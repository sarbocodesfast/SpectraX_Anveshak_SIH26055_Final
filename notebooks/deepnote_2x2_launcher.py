"""Corpus x architecture control -- launcher cell.

This cell deliberately does almost nothing. It fetches the real script from
the repository and starts it under nohup, then returns.

The reason is that ``blockIds`` on ``POST /v2/runs`` is only honoured in live
mode, and a live run dies with its editor session. Doing hours of training
inline would therefore tie the experiment to a browser tab. Detaching means
the run survives a disconnect, and its log can be tailed afterwards.

Everything of substance -- which factors move, which are frozen, and why --
lives in ``notebooks/deepnote_arch_corpus_2x2.py``.
"""

import subprocess
import urllib.request

RAW = ("https://raw.githubusercontent.com/shirish-raj-gupta/SpectraX_Anveshak_SIH26055_Prototype/"
       "main/notebooks/deepnote_arch_corpus_2x2.py")
SCRIPT = "/root/arch_corpus_2x2.py"
LOG = "/work/_jobs/arch2x2.log"

subprocess.run(["mkdir", "-p", "/work/_jobs"], check=False)
urllib.request.urlretrieve(RAW, SCRIPT)
print("fetched", SCRIPT)

# setsid so the job is not in the session's process group and survives it.
proc = subprocess.Popen(
    f"setsid nohup python -u {SCRIPT} > {LOG} 2>&1 < /dev/null &",
    shell=True, start_new_session=True,
)
print(f"launched -> {LOG}  (shell pid {proc.pid})")
print("tail it with:  python scripts/deepnote_ctl.py tail --name arch2x2 --lines 40")

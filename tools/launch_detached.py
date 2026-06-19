"""Launch a script truly detached from Claude Code's bash session
hierarchy. CREATE_BREAKAWAY_FROM_JOB escapes the job object.
Crucially: pass env=os.environ.copy() so PATH propagates (DETACHED
processes don't inherit env by default — this bit roundsP/Q hard).
"""
import os, sys, subprocess
script = sys.argv[1]
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
REPO = "C:/Users/jsmit/source/repos/rl-betfair"
# Capture the wrapper's OWN stdout/stderr (set -e errors, bash-resolution
# failures) so a detached launch that dies is diagnosable — the script itself
# still redirects the python run to its own console log.
dbg = open(os.path.join(REPO, "registry", "launch_detached_debug.log"), "ab")
# Use an explicit Git bash so "bash" can't resolve to WSL (which mishandles a
# Windows cwd + Windows-path .exe and dies silently).
bash_exe = r"C:\Program Files\Git\bin\bash.exe"
prog = bash_exe if os.path.exists(bash_exe) else "bash"
p = subprocess.Popen(
    [prog, script],
    cwd=REPO,
    stdin=subprocess.DEVNULL,
    stdout=dbg,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    creationflags=flags,
    env=os.environ.copy(),  # CRITICAL: propagate PATH
)
print(f"launched pid={p.pid} (bash={prog})")

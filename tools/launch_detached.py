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
p = subprocess.Popen(
    ["bash", script],
    cwd="C:/Users/jsmit/source/repos/rl-betfair",
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
    creationflags=flags,
    env=os.environ.copy(),  # CRITICAL: propagate PATH
)
print(f"launched pid={p.pid}")

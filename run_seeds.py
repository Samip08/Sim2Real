#!/usr/bin/env python3
"""Batch tester: sim + evaluator + solution for many seeds, one after another.

Run from the workspace root (where src/ lives) in a shell that already sourced
/opt/ros/jazzy/setup.bash and install/setup.bash:

    python3 run_seeds.py --tier 1                  # seeds 1..50
    python3 run_seeds.py --tier 2 --start 10 --end 20
    python3 run_seeds.py --tier 3 --seeds 42,456,7
"""
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time

ENV = dict(os.environ, PYTHONUNBUFFERED="1")
KILL_PATTERNS = ["gz sim", "rviz2", "parameter_bridge", "waypoint_node",
                 "encoder_noise_node", "robot_state_publisher", "evaluator.py",
                 "solution.launch.py", "tier1_node", "tier2_node", "tier3_node"]

FIELDS = {
    "collected": r"Waypoints Collected\s*:\s*(\d+)\s*/\s*(\d+)",
    "collisions": r"Collisions Detected\s*:\s*(\d+)",
    "home": r"Returned to Start[^:]*:\s*(YES|NO)",
    "home_err_cm": r"Final Position Error\s*:\s*([\d.]+)\s*cm",
    "odom_m": r"Total Odometry Distance\s*:\s*([\d.]+)\s*m",
    "sim_s": r"Run Duration[^:]*:\s*([\d.]+)\s*s",
}


class Proc:
    def __init__(self, name, cmd, logpath):
        self.name = name
        self.lines = []
        self.log = open(logpath, "w")
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, env=ENV, start_new_session=True)
        self.events = {}
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def watch(self, key, needle):
        ev = threading.Event()
        self.events[key] = (needle, ev)
        if any(needle in l for l in self.lines):
            ev.set()
        return ev

    def _read(self):
        for line in self.p.stdout:
            self.lines.append(line)
            self.log.write(line)
            self.log.flush()
            for needle, ev in self.events.values():
                if needle in line:
                    ev.set()

    def text(self):
        return "".join(self.lines)

    def stop(self, sig=signal.SIGINT, wait=8.0):
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.t.join(timeout=3)
        self.log.close()


def cleanup():
    for pat in KILL_PATTERNS:
        subprocess.run(["pkill", "-9", "-f", pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


COLL_RX = re.compile(r"\[COLLISION\]\s*(.*?)\s+at\s+\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")


def collisions_of(text):
    return [(m.group(1).strip(), float(m.group(2)), float(m.group(3))) for m in COLL_RX.finditer(text)]


def parse(text):
    out = {}
    for k, rx in FIELDS.items():
        m = re.search(rx, text)
        out[k] = m.groups() if m else None
    return out


def run_seed(tier, seed, a, logdir):
    cleanup()
    tag = f"t{tier}_s{seed}"
    sim = Proc("sim", ["ros2", "launch", "postman_pkg", "sim.launch.py", f"tier:={tier}", f"seed:={seed}"]
               + a.sim_args, f"{logdir}/{tag}_sim.log")
    ready = sim.watch("ready", "wp_static_4")
    ready.wait(a.startup)
    time.sleep(a.settle)

    ev = Proc("eval", ["python3", "-u", "src/postman_pkg/scripts/evaluator.py", "--tier", str(tier)],
              f"{logdir}/{tag}_eval.log")
    time.sleep(3)
    sol = Proc("sol", ["ros2", "launch", "postman_solution", "solution.launch.py", f"tier:={tier}"],
               f"{logdir}/{tag}_sol.log")
    done_ev = ev.watch("done", "RUN COMPLETED")
    done_sol = sol.watch("done", "home dwell complete")

    t0 = time.time()
    status = "timeout"
    while time.time() - t0 < a.timeout:
        if done_ev.is_set():
            status = "completed"
            break
        if done_sol.is_set():
            done_ev.wait(a.grace)
            status = "solution_done"
            break
        if ev.p.poll() is not None or sim.p.poll() is not None:
            status = "process_died"
            break
        time.sleep(1)
    wall = time.time() - t0

    ev.stop(signal.SIGINT, wait=10)
    sol.stop()
    sim.stop()
    res = parse(ev.text())
    res.update(seed=seed, status=status, coll_list=collisions_of(ev.text()))
    return res


def res_coll(r):
    out = []
    for part in r["coll_at"].split("; "):
        m = re.match(r"(.*) \(([-\d.]+),([-\d.]+)\)$", part)
        if m:
            out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


def flat(res):
    c = res["collected"]
    return {
        "seed": res["seed"], "status": res["status"],
        "collected": f"{c[0]}/{c[1]}" if c else "?",
        "collisions": res["collisions"][0] if res["collisions"] else "?",
        "home": res["home"][0] if res["home"] else "?",
        "home_err_cm": res["home_err_cm"][0] if res["home_err_cm"] else "?",
        "coll_at": "; ".join(f"{k} ({x:.2f},{y:.2f})" for k, x, y in res["coll_list"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--seeds", type=str, default="", help="comma list, overrides start/end")
    ap.add_argument("--timeout", type=int, default=1500, help="wall seconds per seed")
    ap.add_argument("--startup", type=int, default=90, help="max wall s to wait for sim spawn")
    ap.add_argument("--settle", type=int, default=3)
    ap.add_argument("--grace", type=int, default=15)
    ap.add_argument("--sim-args", nargs="*", default=[], help="extra sim.launch.py args, e.g. headless:=true")
    ap.add_argument("--logdir", default="seed_logs")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(range(a.start, a.end + 1))
    os.makedirs(a.logdir, exist_ok=True)
    csvpath = f"seed_results_tier{a.tier}.csv"
    rows = []
    try:
        with open(csvpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "status", "collected", "collisions", "home",
                                              "home_err_cm", "coll_at"])
            w.writeheader()
            for s in seeds:
                r = flat(run_seed(a.tier, s, a, a.logdir))
                rows.append(r)
                w.writerow(r)
                f.flush()
                print(f"seed {s:>3}: {r['collected']:>4} collected | coll {r['collisions']} | "
                      f"home {r['home']} ({r['home_err_cm']} cm) | {r['status']}", flush=True)
                for kind, cx, cy in res_coll(r):
                    kind = re.sub(r"\s*collision\s*$", "", kind)
                    print(f"           ! {kind} collision at ({cx:.2f}, {cy:.2f})", flush=True)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cleanup()

    if rows:
        full = [r for r in rows if r["collected"] == "7/7" and r["home"] == "YES"]
        clean = [r for r in full if str(r["collisions"]) == "0"]
        print(f"\n{len(rows)} seeds | full success (7/7 + home): {len(full)} | also zero collisions: {len(clean)}")
        bad = [r["seed"] for r in rows if r not in full]
        if bad:
            print("failed seeds:", bad)
        spots = [(r["seed"], res_coll(r)) for r in rows if res_coll(r)]
        if spots:
            print("collisions by seed:")
            for sd, lst in spots:
                print(f"  seed {sd}: " + ", ".join(f"({x:.2f},{y:.2f})" for _, x, y in lst))
        print(f"results: {csvpath}  logs: {a.logdir}/")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
mission_runner.py
Usage: python3 mission_runner.py
Requirements: pymoos installed, pAntler and MOOS tools in PATH
Place templates in templates/alpha_template.moos and templates/alpha_bhv_template.bhv
"""

import os, shutil, subprocess, time, threading, csv, glob, random, signal
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# If you want to use pymoos for reliable MISSION detection:
try:
    import pymoos
    HAVE_PYMOOS = True
except Exception:
    HAVE_PYMOOS = False
    print("pymoos not available — script will fallback to logfile polling (slower).")

# --- CONFIG (edit as needed) ---
TEMPLATE_DIR = "templates"
RUNS_DIR = "runs"
DATASET_DIR = "dataset"
NUM_RUNS = 5000
MOOS_HOST = "localhost"
MOOS_PORT = 9000
MISSION_TIMEOUT = 300  # seconds per run (safety)
PAUSE_BETWEEN_RUNS = 2  # seconds
# default param ranges
START_X_RANGE = (-20, 20)
START_Y_RANGE = (-60, 60)
HEADING_RANGE = (0, 359)
DRIFT_ANGLE_RANGE = (0, 359)
DRIFT_SPEED_RANGE = (0.0, 2.0)
# pid variation ranges (multipliers around baseline)
YAW_KP_BASE = 0.9
YAW_KI_BASE = 0.3
YAW_KD_BASE = 0.3
SPD_KP_BASE = 1.0
SPD_KI_BASE = 0.0
SPD_KD_BASE = 0.0

# default fixed values
SPEED_FACTOR = 0  # if you want to use speed_pid instead set to 0

# a simple waypoint generator (straight line of 3 points)
def generate_waypoints_random(num_points=3, x_range=(-100, 100), y_range=(-100, 100)):
    pts = []
    for _ in range(num_points):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        pts.append(f"{x:.1f},{y:.1f}")
    return " : ".join(pts)

# helper: load template
def load_template(name):
    p = os.path.join(TEMPLATE_DIR, name)
    with open(p, "r") as f:
        return f.read()

# helper: write populated mission files
def write_mission_files(run_dir, moos_txt, bhv_txt):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "alpha.moos"), "w") as f:
        f.write(moos_txt)
    with open(os.path.join(run_dir, "alpha.bhv"), "w") as f:
        f.write(bhv_txt)

def read_alog(logfile):
    df = pd.read_csv(logfile, sep=r'\s{2,}', engine='python', comment='%', header=None, names=["time", "var", "source", "value"])

    df = df[df["var"].isin(["NAV_X", "NAV_Y"])]
    df_pivot = df.pivot(index="time", columns="var", values="value").reset_index()
    df_pivot = df_pivot.sort_values("time")
    df_pivot["NAV_X"] = df_pivot["NAV_X"].astype(float)
    df_pivot["NAV_Y"] = df_pivot["NAV_Y"].astype(float)
    return df_pivot

def draw_trajectory(logfile, outfile, show_waypoints):
    waypoints_list = show_waypoints.split(" : ")
    waypoints = [list(map(float, wp.split(","))) for wp in waypoints_list]
    df = read_alog(logfile)

    xs = df["NAV_X"].values
    ys = df["NAV_Y"].values

    plt.figure(figsize=(6,6), dpi=100)
    plt.plot(xs, ys, color="white", linewidth=2)
    plt.scatter(xs[0], ys[0], color="green", s=50, label="Start")
    plt.scatter(xs[-1], ys[-1], color="red", s=50, label="Koniec")

    for (wx, wy) in waypoints:
        plt.scatter(wx, wy, color="yellow", marker="x", s=70)

    plt.xlim(-120, 120)
    plt.ylim(-120,120)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.axis("off")  # bez osi
    plt.gca().set_facecolor("black")  # tło czarne
    plt.savefig(outfile, facecolor='black', bbox_inches="tight", pad_inches=0)
    plt.close()

# find alog file produced in the run_dir (newest)
def find_alog(run_dir, wait_seconds=2):
    # pLogger writes files like <community>.<something>.alog; search for *.alog
    log_dir = os.path.join(run_dir, "MOOSLog*")
    files = glob.glob(os.path.join(log_dir, "*.alog"))
    if files:
        # return newest
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    # sometimes pAntler writes in parent; check shortly after
    time.sleep(wait_seconds)
    files = glob.glob(os.path.join(run_dir, "*.alog"))
    return files[0] if files else None

# monitor MISSION using pymoos (preferred)
def wait_for_mission_complete_pymoos(timeout, run_dir, host=MOOS_HOST, port=MOOS_PORT):
    client_name = f"runner_{int(time.time()*1000)%1000000}"

    comms = pymoos.comms()
    mission_event = threading.Event()
    start_time = time.time()

    def on_mail():
        for m in comms.fetch():
            try:
                key = m.key()
            except Exception:
                continue
            try:
                val = m.string()
            except:
                val = None
            if key == "MISSION" and val and val.strip().lower() == "complete":
                mission_event.set()
            if key == "RETURN" and val and val.strip().lower() in ("true","1"):
                mission_event.set()
        return True

    def on_connect():
        # rejestrujemy subskrypcje dopiero po nawiązaniu połączenia
        comms.register("MISSION", 0)
        comms.register("RETURN", 0)
        return True

    comms.set_on_connect_callback(on_connect)
    comms.set_on_mail_callback(on_mail)

    # TU jest sedno: podaj też nazwę klienta
    if not comms.run(host, int(port), client_name):
        print("pymoos: cannot connect to MOOSDB")
        return False, None

    finished = mission_event.wait(timeout=float(timeout))
    comms.close(True)  # zamknij połączenie i wątki pymoos

    elapsed = time.time() - start_time
    return finished, elapsed    

# fallback: check alog content until 'MISSION=complete'
def wait_for_mission_complete_by_log(run_dir, timeout):
    start = time.time()
    alog_path = None
    print(f"[DEBUG] Start monitoring logs for run_dir: {run_dir}")
    while time.time() - start < timeout:
        # find alog and tail last lines
        alog_path = find_alog(run_dir, wait_seconds=0.1)
        if alog_path:
            with open(alog_path, "r", errors="ignore") as f:
                content = f.read()
                if "MISSION" in content:
                    # naive search for 'MISSION' and 'complete'
                    if "MISSION" in content and "complete" in content:
                        print(f"[DEBUG] Mission complete detected after {elapsed:.1f}s")
                        return True, time.time()-start, alog_path
        time.sleep(1.0)
    print(f"[DEBUG] Timeout reached ({elapsed:.1f}s), mission not complete")
    return False, time.time()-start, alog_path

# run one mission (generate files, run pAntler, wait)
def run_one(run_id, params):
    run_dir = os.path.join(RUNS_DIR, f"run_{run_id:04d}")
    os.makedirs(run_dir, exist_ok=True)

    # load templates
    moos_tpl = load_template("alpha_template.moos")
    bhv_tpl  = load_template("alpha_bhv_template.bhv")

    waypoints_str = params.get("WAYPOINTS") or generate_waypoints_random()

    moos_filled = moos_tpl.replace("{{RUN_ID}}", str(run_id)) \
                         .replace("{{START_X}}", f"{params['START_X']:.3f}") \
                         .replace("{{START_Y}}", f"{params['START_Y']:.3f}") \
                         .replace("{{START_HEADING}}", f"{params['START_HEADING']:.1f}") \
                         .replace("{{DRIFT_ANGLE}}", f"{params['DRIFT_ANGLE']:.1f}") \
                         .replace("{{DRIFT_SPEED}}", f"{params['DRIFT_SPEED']:.3f}") \
                         .replace("{{WAYPOINTS}}", waypoints_str) \
                         .replace("{{YAW_KP}}", f"{params['YAW_KP']:.4f}") \
                         .replace("{{YAW_KI}}", f"{params['YAW_KI']:.4f}") \
                         .replace("{{YAW_KD}}", f"{params['YAW_KD']:.4f}") \
                         .replace("{{YAW_ILIM}}", f"{params['YAW_ILIM']:.4f}") \
                         .replace("{{SPD_KP}}", f"{params['SPD_KP']:.4f}") \
                         .replace("{{SPD_KI}}", f"{params['SPD_KI']:.4f}") \
                         .replace("{{SPD_KD}}", f"{params['SPD_KD']:.4f}") \
                         .replace("{{SPD_ILIM}}", f"{params['SPD_ILIM']:.4f}") \
                         .replace("{{SPEED_FACTOR}}", f"{params['SPEED_FACTOR']}") \
                         .replace("{{MAXRUDDER}}", f"{params['MAXRUDDER']}") \
                         .replace("{{MAXTHRUST}}", f"{params['MAXTHRUST']}")

    bhv_filled = bhv_tpl.replace("{{WAYPOINTS}}", waypoints_str).replace("{{SPEED}}", f"{params['SPEED']}")

    write_mission_files(run_dir, moos_filled, bhv_filled)

    # kill processes of previous run if something went wrong and didn't kill all processes
    subprocess.run(["killall", "pAntler", "MOOSDB", "pLogger", "uSimMarineV22", "pMarinePIDV22", "pHelmIvP", "pMarineViewer", "pNodeReporter", "uTimerScript"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    # start pAntler
    p = subprocess.Popen(["pAntler", "alpha.moos"], cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid, text=True)
    print(f"[run {run_id}] started pAntler (pid={p.pid}), params: start=({params['START_X']:.1f},{params['START_Y']:.1f}) drift={params['DRIFT_ANGLE']:.1f},{params['DRIFT_SPEED']:.2f}")

    mission_finished = False
    elapsed = None

    if HAVE_PYMOOS:
        # wait using pymoos
        mission_finished, elapsed = wait_for_mission_complete_pymoos(MISSION_TIMEOUT, run_dir)
    else:
        # fallback: monitor log
        mission_finished, elapsed, alog_path = wait_for_mission_complete_by_log(run_dir, MISSION_TIMEOUT)

    # after wait, terminate pAntler if still running
    try:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
    except Exception as e:
        print("Error terminating pAntler:", e)

    # find .alog and move to dataset
    alog = find_alog(run_dir, wait_seconds=0.5)
    out_dir = os.path.join(DATASET_DIR, f"run_{run_id:04d}")
    os.makedirs(out_dir, exist_ok=True)
    if alog:
        shutil.move(alog, os.path.join(out_dir, os.path.basename(alog)))
        print(f"[run {run_id}] moved alog -> {out_dir}")
    else:
        print(f"[run {run_id}] WARNING: no alog found in {run_dir}")

    # capture stdout/stderr for debug
    try:
        out, err = p.communicate(timeout=0.1)
    except Exception:
        out, err = "", ""
    with open(os.path.join(out_dir, "pAntler.stdout.txt"), "w") as f:
        f.write(out or "")
    with open(os.path.join(out_dir, "pAntler.stderr.txt"), "w") as f:
        f.write(err or "")

    # create trajectory image in dataset
    draw_trajectory(os.path.join(out_dir, os.path.basename(alog)), out_dir + "/traj.png", waypoints_str)

    return mission_finished, elapsed, out_dir

# main loop: generate random params and run
def main():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # load templates early to fail fast
    _ = load_template("alpha_template.moos")
    _ = load_template("alpha_bhv_template.bhv")

    manifest_path = os.path.join(DATASET_DIR, "manifest.csv")
    # write header if not exists
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(["run_id","start_x","start_y","start_heading","drift_angle","drift_speed",
                             "yaw_kp","yaw_ki","yaw_kd","yaw_ilim",
                             "spd_kp","spd_ki","spd_kd","spd_ilim",
                             "speed","waypoints","status","elapsed","timestamp","traj_img"
                             ])
        start_run = 1
    else:
        with open(manifest_path, "r") as mf:
            done = sum(1 for _ in mf) - 1
        start_run = done + 1

    i = start_run

    while i < NUM_RUNS:
        # draw random params (you can replace with deterministic scan)
        start_x = random.uniform(*START_X_RANGE)
        start_y = random.uniform(*START_Y_RANGE)
        heading = random.uniform(*HEADING_RANGE)
        drift_angle = random.uniform(*DRIFT_ANGLE_RANGE)
        drift_speed = random.uniform(*DRIFT_SPEED_RANGE)

        # small randomization of pid around base
        yaw_kp = YAW_KP_BASE * random.uniform(0.4, 1.8)
        yaw_ki = YAW_KI_BASE * random.uniform(0.0, 1.5)
        yaw_kd = YAW_KD_BASE * random.uniform(0.0, 1.5)
        yaw_ilim = random.uniform(0.03, 0.2)

        spd_kp = SPD_KP_BASE * random.uniform(0.7, 1.3)
        spd_ki = SPD_KI_BASE * random.uniform(0.0, 1.0)
        spd_kd = SPD_KD_BASE * random.uniform(0.0, 1.0)
        spd_ilim = 0.07

        maxrudder = 100
        maxthrust = 100
        desired_speed = random.uniform(2.8, 4.0)

        waypoints = generate_waypoints_random(num_points=3, x_range=(-100, 100), y_range=(-100, 100))

        params = {
            "START_X": start_x, "START_Y": start_y, "START_HEADING": heading,
            "DRIFT_ANGLE": drift_angle, "DRIFT_SPEED": drift_speed,
            "WAYPOINTS": waypoints,
            "YAW_KP": yaw_kp, "YAW_KI": yaw_ki, "YAW_KD": yaw_kd, "YAW_ILIM": yaw_ilim,
            "SPD_KP": spd_kp, "SPD_KI": spd_ki, "SPD_KD": spd_kd, "SPD_ILIM": spd_ilim,
            "SPEED_FACTOR": SPEED_FACTOR, "MAXRUDDER": maxrudder, "MAXTHRUST": maxthrust,
            "SPEED": desired_speed
        }

        ok, elapsed, out_dir = run_one(i, params)
        status = "complete" if ok else "timeout_or_error"

        if not ok:
            print(f"[run {i}] ERROR: ({status}), deleting run and restart")
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(RUNS_DIR, f"run_{i:04d}"), ignore_errors=True)
            time.sleep(PAUSE_BETWEEN_RUNS)
            continue  # nie inkrementujemy i, run będzie powtórzony

        traj_img = out_dir + "/traj.png"
        timestamp = datetime.utcnow().isoformat()

        with open(manifest_path, "a", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow([i,
                             f"{start_x:.3f}", f"{start_y:.3f}", f"{heading:.1f}",
                             f"{drift_angle:.1f}", f"{drift_speed:.3f}",
                             f"{yaw_kp:.4f}", f"{yaw_ki:.4f}", f"{yaw_kd:.4f}", f"{yaw_ilim:.4f}",
                             f"{spd_kp:.4f}", f"{spd_ki:.4f}", f"{spd_kd:.4f}", f"{spd_ilim:.4f}",
                             f"{desired_speed:.4f}", waypoints, status, f"{elapsed:.1f}", timestamp, traj_img])

        print(f"[run {i}] status={status}, elapsed={elapsed:.1f}s -> saved to manifest")
        time.sleep(PAUSE_BETWEEN_RUNS)
        i += 1

if __name__ == "__main__":
    main()
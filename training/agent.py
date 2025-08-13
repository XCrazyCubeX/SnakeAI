import os

# ---- make everything headless BEFORE importing things that may touch X ----
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")            # matplotlib headless
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen") # Qt headless (just in case)

from multiprocessing import get_context
import torch
from stable_baselines3 import PPO
from env import Snake  # ensure this respects headless (no display init if not needed)

# Paths
MODELS_DIR = "models/PPO"
LOGS_DIR   = "logs/PPO"

def train_model(process_num: int, best_process: str, best_model: str):
    # device per worker (no CUDA touch at import time)
    if torch.cuda.is_available():
        torch.cuda.set_device(process_num % torch.cuda.device_count())
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    # lazy import for plotting (headless)
    import matplotlib.pyplot as plt
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # each process its own log dir
    log_dir_process = os.path.join(LOGS_DIR, f"process_{process_num}")
    os.makedirs(log_dir_process, exist_ok=True)

    # build env HEADLESS (no render)
    env = Snake()   # make sure your env does NOT open a pygame window during training
    env.reset()

    total_steps = 50_000
    count = 0

    while True:
        # start fresh or load
        if count == 0:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device=device,
                tensorboard_log=log_dir_process,  # only works on fresh init
            )
        else:
            load_path = f"{MODELS_DIR}/process_{best_process}_model_{best_model}"
            # NOTE: PPO.load does NOT accept tensorboard_log/verbose kwargs
            model = PPO.load(load_path, env=env, device=device, print_system_info=False)
            print(f"[p{process_num}] Loading model {load_path}")

        # train
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO")

        # save
        count += 1
        save_path = f"{MODELS_DIR}/process_{process_num}_model_{count}"
        model.save(save_path)

        # update pointers (in-memory only here; if you want cross-proc, write to a small file)
        best_process = process_num
        best_model = count

        # optional: quick heartbeat plot (stays headless)
        try:
            plt.figure()
            plt.plot([0, count])
            plt.title(f"p{process_num} iters")
            plt.savefig(os.path.join(log_dir_process, f"heartbeat_{count}.png"))
            plt.close()
        except Exception:
            pass

    # env.close()  # unreachable in while True

def main():
    best_process = input("Please select best performing process: ").strip()
    best_model   = input("Please select best performing model: ").strip()

    num_processes = 2

    # IMPORTANT: use spawn so no X connection is inherited
    ctx = get_context("spawn")
    procs = []
    for i in range(num_processes):
        p = ctx.Process(target=train_model, args=(i, best_process, best_model), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == "__main__":
    # For PyTorch specifically, force spawn as well (no-op if already spawn)
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # Optional: tame CPU thread explosions (more stable multi-proc)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    main()

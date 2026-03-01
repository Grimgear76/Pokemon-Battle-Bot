import asyncio
import multiprocessing

from bot_training import train_new, train_continue, eval_model, play_vs_human, train_vs_opponent

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # -----------------------------
    # Configuration
    # -----------------------------
    # Modes: "new" | "continue" | "eval" | "human" | "league"
    MODE = "eval"
    MODEL_NAME     = "ParallelTest13"
    TRAINING_STEPS = 200000

    GEN         = 2          # 1 or 2 — controls battle format passed to training functions
    N_ENVS_RUN  = 6
    USE_SUBPROC = True

    # Human mode
    HUMAN_USERNAME  = "Grimgear76"
    N_HUMAN_BATTLES = 3

    # League mode — learner trains against frozen opponent
    # Ladder example:
    #   new model name (Learner)
    #   LEARNER_NAME = "Gen2"  |  OPPONENT_NAME = "ParallelTest8"
    #   next iteration
    #   LEARNER_NAME = "Gen3"  |  OPPONENT_NAME = "Gen2"
    LEARNER_NAME  = "Test8Gen1"
    OPPONENT_NAME = "ParallelTest8"

    # Derive battle format from GEN
    BATTLE_FORMAT = f"gen{GEN}randombattle"

    # -----------------------------
    # Run
    # -----------------------------
    if MODE == "new":
        train_new(MODEL_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT)

    elif MODE == "continue":
        train_continue(MODEL_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT)

    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=200, battle_format=BATTLE_FORMAT)

    elif MODE == "human":
        asyncio.run(play_vs_human(MODEL_NAME, HUMAN_USERNAME, N_HUMAN_BATTLES, battle_format=BATTLE_FORMAT))

    elif MODE == "league":
        train_vs_opponent(LEARNER_NAME, OPPONENT_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT)


# -----------------------------
# Notes
# -----------------------------
# Tensorboard:   tensorboard --logdir ./tensorboard_logs/
# Activate venv: .\.venv\Scripts\Activate.ps1
#
# Local Showdown setup:
#   git clone https://github.com/smogon/pokemon-showdown.git
#   cd pokemon-showdown && npm install
#   cp config/config-example.js config/config.js
#   node pokemon-showdown start --no-security
#
# If npm install fails for pg:
#   Stop-Process -Name "node" -ErrorAction SilentlyContinue
#   npm install pg --save-dev
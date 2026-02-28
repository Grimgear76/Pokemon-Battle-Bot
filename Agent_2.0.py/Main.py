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
    MODEL_NAME     = "ParallelTest10"
    TRAINING_STEPS = 250000

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

    # -----------------------------
    # Run
    # -----------------------------
    if MODE == "new":
        train_new(MODEL_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)

    elif MODE == "continue":
        train_continue(MODEL_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)

    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=100)

    elif MODE == "human":
        asyncio.run(play_vs_human(MODEL_NAME, HUMAN_USERNAME, N_HUMAN_BATTLES))

    elif MODE == "league":
        train_vs_opponent(LEARNER_NAME, OPPONENT_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC)


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
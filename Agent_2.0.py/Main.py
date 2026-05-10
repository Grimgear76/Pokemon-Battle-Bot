import asyncio
import multiprocessing

from bot_training import train_new, train_continue, eval_model, eval_model_vs_model, play_vs_human, train_vs_opponent, train_self_play

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Modes: "new" | "continue" | "eval" | "eval_vs" | "human" | "league" | "self_play"
    MODE = "eval"
    MODEL_NAME     = "Test"
    TRAINING_STEPS = 1000000

    EVAL_OPPONENT     = "heuristic"   # "max" | "heuristic" | "random"
    CONTINUE_OPPONENT = "heuristic"   # "maxdamage" | "heuristic" | "mix"

    GEN         = 2
    N_ENVS_RUN  = 8
    USE_SUBPROC = True

    HUMAN_USERNAME  = "Grimgear76"
    N_HUMAN_BATTLES = 3

    LEARNER_NAME  = "Agent23"
    OPPONENT_NAME = "Agent23Gen2"

    CHALLENGER_NAME  = "Agent23"
    EVAL_VS_OPPONENT = "Agent23Gen2"
    N_EVAL_BATTLES   = 500

    SELF_PLAY_LEARNER = "Agent29"
    # Pool entries must share the same OBS_SIZE as the learner (839-dim for Agent29+).
    SELF_PLAY_POOL    = ["heuristic", "heuristic", "heuristic", "heuristic", "max", "random"]

    BATTLE_FORMAT = f"gen{GEN}randombattle"


    if MODE == "new":
        train_new(MODEL_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT)

    elif MODE == "continue":
        train_continue(MODEL_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT, opponent=CONTINUE_OPPONENT)

    elif MODE == "eval":
        eval_model(MODEL_NAME, n_battles=N_EVAL_BATTLES, battle_format=BATTLE_FORMAT, opponent=EVAL_OPPONENT)

    elif MODE == "eval_vs":
        eval_model_vs_model(CHALLENGER_NAME, EVAL_VS_OPPONENT, n_battles=N_EVAL_BATTLES, battle_format=BATTLE_FORMAT)

    elif MODE == "human":
        asyncio.run(play_vs_human(MODEL_NAME, HUMAN_USERNAME, N_HUMAN_BATTLES, battle_format=BATTLE_FORMAT))

    elif MODE == "league":
        train_vs_opponent(LEARNER_NAME, OPPONENT_NAME, TRAINING_STEPS, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT)

    elif MODE == "self_play":
        train_self_play(SELF_PLAY_LEARNER, TRAINING_STEPS, SELF_PLAY_POOL, n_envs=N_ENVS_RUN, use_subproc=USE_SUBPROC, battle_format=BATTLE_FORMAT, seed_from="Agent29")


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
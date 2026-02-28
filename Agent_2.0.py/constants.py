from pathlib import Path

# -----------------------------
# Server Configuration
# -----------------------------
# LOCAL:    Use LocalhostServerConfiguration (requires local showdown server running)
# SHOWDOWN: Use ShowdownServerConfiguration  (requires registered account)
from poke_env import LocalhostServerConfiguration, ShowdownServerConfiguration
SERVER_CONFIG = LocalhostServerConfiguration

# -----------------------------
# Directories and constants
# -----------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# OBS_SIZE breakdown (90 total):
#   moves_base_power        4
#   moves_dmg_multiplier    4
#   moves_pp_ratio          4
#   self_team_status        6
#   opponent_team_status    6
#   team_hp_ratio           6
#   opponent_hp_ratio       6
#   team_identifier         6
#   opponent_identifier     6
#   self_status             6
#   opponent_status         6
#   special_case            2
#   active_features         5
#   opp_active_features     5
#   own_speed               1
#   opp_speed               1
#   opp_move_eff            4
#   own_atk_boost           1
#   own_spe_boost           1
#   own_status_flags        5  [slp, frz, par, brn, psn]
#   opp_status_flags        5  [slp, frz, par, brn, psn]
OBS_SIZE = 90

ACTION_SPACE_SIZE = 11
NET_ARCH = [256, 128, 64]
LEARNING_RATE = 1e-4
MAX_SPEED = 140

# --- Parallel training config ---
N_ENVS = 4
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10
ENT_COEF = 0.02
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95

STATUS_MAP = {
    None: 0.0,
    "brn": 0.2,
    "frz": 0.4,
    "par": 0.6,
    "psn": 0.8,
    "tox": 0.8,
    "slp": 1.0,
}

def model_path(name) -> Path:
    return MODEL_DIR / f"{name}.zip"
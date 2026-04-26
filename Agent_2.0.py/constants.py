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

# OBS_SIZE breakdown (165 total):
#   moves_base_power        4
#   moves_dmg_multiplier    4
#   moves_pp_ratio          4
#   team_hp_ratio           6
#   opponent_hp_ratio       6
#   team_identifier         6
#   opponent_identifier     6
#   self_status             6   status scalar per bench slot (name.lower() -> STATUS_MAP)
#   opponent_status         6   status scalar per bench slot
#   special_case            2
#   active_features        37   [hp(1), type1_onehot(18), type2_onehot(18)]
#   opp_active_features    37   [hp(1), type1_onehot(18), type2_onehot(18)]
#   own_speed               1
#   opp_speed               1
#   opp_move_eff            4
#   own_boosts              5   [atk, def, spe, spa, spd]
#   opp_boosts              5   [atk, def, spe, spa, spd]
#   own_item_id             1
#   opp_item_id             1
#   own_status_flags        5   [slp, frz, par, brn, psn]  binary, active mon only
#   opp_status_flags        5   [slp, frz, par, brn, psn]  binary, active mon only
#   turn_counter            1   normalized [0, 1] over 150 turns
#   own_alive_flags         6   explicit binary: 1.0=alive, 0.0=fainted
#   opp_alive_flags         6   explicit binary: 1.0=alive, 0.0=fainted
#                         165
OBS_SIZE = 165

ACTION_SPACE_SIZE = 11

# Wider, more balanced network 
NET_ARCH = [256, 256, 128]

# FIX: Electrode's base Speed in Gen 2 is 150
MAX_SPEED = 150

LEARNING_RATE = 1e-4

# --- Parallel training config ---
N_ENVS = 4
N_STEPS = 2048        
BATCH_SIZE = 512
N_EPOCHS = 10
ENT_COEF = 0.04
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95

STATUS_MAP = {
    None:   0.0,
    "brn":  0.2,
    "frz":  0.4,
    "par":  0.6,
    "psn":  0.8,
    "tox":  0.8,
    "slp":  1.0,
}

# Species ID encoding — uses the built-in Pokédex `num` field (1–251 for Gen 1+2).
# ID is already a meaningful ordinal (National Dex number).
# Normalisation: (num / 251) * 2 - 1  maps  #1 -> ~-0.992,  #251 -> 1.0
# Unknown species (not in pokedex) fall back to 0.0 (mid-range, neutral).
SPECIES_NUM_SCALE = 251.0  # Gen 2 National Dex number


def model_path(name) -> Path:
    return MODEL_DIR / f"{name}.zip"
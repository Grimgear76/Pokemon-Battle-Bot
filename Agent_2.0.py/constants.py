from pathlib import Path

# LOCAL: LocalhostServerConfiguration | SHOWDOWN: ShowdownServerConfiguration
from poke_env import LocalhostServerConfiguration, ShowdownServerConfiguration
SERVER_CONFIG = LocalhostServerConfiguration

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)  # create on import so saves never fail

# OBS_SIZE layout (839 total):
#   moves_base_power        4
#   moves_dmg_multiplier    4
#   moves_pp_ratio          4
#   team_hp_ratio           6   insertion order, matches switch action slots 0-5
#   opponent_hp_ratio       6   discovery order
#   team_identifier         6
#   opponent_identifier     6
#   self_status             6
#   opponent_status         6
#   special_case            2
#   active_features        37   [hp(1), type1_onehot(18), type2_onehot(18)]
#   opp_active_features    37
#   own_speed               1
#   opp_speed               1
#   opp_move_eff            4
#   own_boosts              5   [atk, def, spe, spa, spd]
#   opp_boosts              5
#   own_status_flags        5   [slp, frz, par, brn, psn]
#   opp_status_flags        5
#   turn_counter            1   normalized [0, 1] over 150 turns
#   own_alive_flags         6
#   opp_alive_flags         6   1.0 only for SEEN-and-alive
#   bench_off_multiplier    6   best offensive mult vs opp active type per own slot
#   bench_def_vulnerability 6   worst seen-opp-move mult vs each own slot's types
#   matchup_features        3   [own_off, opp_off, diff] for current active matchup
#   weather                 4   [sun, rain, sand, hail] one-hot
#   own_side_conditions     5   [spikes, reflect, light_screen, safeguard, mist]
#   opp_side_conditions     5
#   own_slot_features     252   6 slots × 42 dims [type1_onehot 18, type2_onehot 18, base_stats 6]
#   opp_slot_features     252   zeros for unseen slots
#   opp_seen_flags          6   1.0 for revealed opp slots
#   own_is_active           6   one-hot: which own slot is active
#   opp_is_active           6   one-hot: which seen-opp slot is active
#   force_switch_flag       1
#   speed_advantage_flag    1   +1/0/-1
#   moves_type_onehot      72   4 own moves × 18-dim type one-hot
#   sleep_counter_own       1   turns slept / 7
#   sleep_counter_opp       1
#   toxic_counter_own       1   toxic stacks / 15
#   toxic_counter_opp       1
#   substitute_own          1
#   substitute_opp          1
#   own_action_history     33   last 3 actions × 11-way one-hot (newest first)
#   own_move_categories    12   4 moves × 3-dim [physical, special, status]
#                         839
OBS_SIZE = 839

# 11 actions: slots 0-5 = switch to team member, slots 6-9 = use move 1-4, slot 10 = default/struggle
ACTION_SPACE_SIZE = 11
ACTION_HISTORY_LEN = 3  # last N actions stored in obs; helps detect loops
N_MOVE_CATEGORIES  = 3  # physical / special / status

# Toggle between the slot-equivariant feature extractor and a plain MLP.
USE_SLOT_EQUIVARIANT = True

# Slot block layout — must stay in sync with embed_battle_impl.
N_SLOTS = 6
SLOT_DIM = 42                        # 18 type1 + 18 type2 + 6 base_stats
OWN_SLOT_OFFSET = 192
OPP_SLOT_OFFSET = 444                # = OWN_SLOT_OFFSET + SLOT_BLOCK_LEN
SLOT_BLOCK_LEN = N_SLOTS * SLOT_DIM  # 252
SLOT_HIDDEN = 32                     # per-slot embedding size

# Wider pi for policy capacity; separate vf avoids shared-bottleneck gradient conflict.
NET_ARCH = {"pi": [1024, 512, 256], "vf": [512, 256, 128]}

MAX_SPEED = 150  # Electrode Gen 2 base Speed — used to normalize speed to [-1, 1]

LEARNING_RATE = 5e-5

# PPO hyperparameters — effective batch = N_STEPS * N_ENVS (16 384 steps per update).
N_ENVS = 4
N_STEPS = 4096
BATCH_SIZE = 512
N_EPOCHS = 8
ENT_COEF = 0.008
VF_COEF = 1.0       # raised from 0.5; value function errors dominate early training
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.92
TARGET_KL = 0.02    # early-stop rollout if KL exceeds this; prevents policy collapse
GAMMA = 0.995       # high discount emphasizes long-term HP preservation

# Ordinal encoding for status conditions — higher = more disabling.
STATUS_MAP = {
    None:   0.0,
    "brn":  0.2,
    "frz":  0.4,
    "par":  0.6,
    "psn":  0.8,
    "tox":  0.8,
    "slp":  1.0,
}

SPECIES_NUM_SCALE = 251.0  # Gen 2 National Dex max — used to normalize dex numbers to [-1, 1]


def model_path(name) -> Path:
    """Returns the .zip save path for a named model checkpoint."""
    return MODEL_DIR / f"{name}.zip"

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

# OBS_SIZE breakdown (794 total):
#   moves_base_power        4
#   moves_dmg_multiplier    4
#   moves_pp_ratio          4
#   team_hp_ratio           6   insertion order, matches action slots 0-5
#   opponent_hp_ratio       6   insertion order (discovery order on opp side)
#   team_identifier         6   insertion order, species_num scalar (kept for stable id signal)
#   opponent_identifier     6   discovery order
#   self_status             6   insertion order, STATUS_MAP scalar per slot
#   opponent_status         6   discovery order
#   special_case            2
#   active_features        37   [hp(1), type1_onehot(18), type2_onehot(18)]
#   opp_active_features    37   [hp(1), type1_onehot(18), type2_onehot(18)]
#   own_speed               1
#   opp_speed               1
#   opp_move_eff            4
#   own_boosts              5   [atk, def, spe, spa, spd]
#   opp_boosts              5   [atk, def, spe, spa, spd]
#   own_status_flags        5   [slp, frz, par, brn, psn]  binary, active mon only
#   opp_status_flags        5   [slp, frz, par, brn, psn]  binary, active mon only
#   turn_counter            1   normalized [0, 1] over 150 turns
#   own_alive_flags         6   insertion order, explicit binary (1 if seen-and-alive)
#   opp_alive_flags         6   discovery order, 1 only for SEEN-and-alive (unseen → 0)
#   bench_off_multiplier    6   per-team-slot best offensive mult vs opp active type
#   bench_def_vulnerability 6   per-team-slot worst seen-opp-move mult vs that mon's types
#   matchup_features        3   [own_off, opp_off, diff] for current active matchup
#   weather                 4   [sun, rain, sand, hail] one-hot
#   own_side_conditions     5   [spikes, reflect, light_screen, safeguard, mist] binary (Agent23: +Safeguard +Mist)
#   opp_side_conditions     5   [spikes, reflect, light_screen, safeguard, mist] binary (Agent23: +Safeguard +Mist)
#   own_slot_features     252   6 slots × 42 dims [type1_onehot 18, type2_onehot 18, base_stats 6]
#   opp_slot_features     252   6 slots × 42 dims (zeros for unseen slots)
#   opp_seen_flags          6   1.0 for slots holding a seen opp mon, 0.0 otherwise
#   own_is_active           6   one-hot: which own slot is the current active mon
#   opp_is_active           6   one-hot: which seen-opp slot is the current active mon
#   force_switch_flag       1   1.0 when battle.force_switch is True
#   speed_advantage_flag    1   +1/0/-1 for own_eff_speed vs opp_eff_speed
#   moves_type_onehot      72   4 own moves × 18-dim type one-hot (Agent21)
#   sleep_counter_own       1   own active sleep turns / 7 (Agent23 — 0 if not asleep)
#   sleep_counter_opp       1   opp active sleep turns / 7 (Agent23)
#   toxic_counter_own       1   own active toxic stacks / 15 (Agent23 — 0 if not TOX)
#   toxic_counter_opp       1   opp active toxic stacks / 15 (Agent23)
#   substitute_own          1   1.0 if own active is behind a sub (Agent23)
#   substitute_opp          1   1.0 if opp active is behind a sub (Agent23)
#                         794
# Removed: own_item_id, opp_item_id — Gen 2 randombattle uses Leftovers exclusively.
# Added (Agent12): bench_off_multiplier + bench_def_vulnerability — direct bench-matchup signal.
# Added (Agent13): matchup_features (current active off/def signal — fixes the no-switch /
#   stay-in-losing-matchup behaviour seen in Agent12 battle logs e.g. Quagsire-vs-Kingdra
#   and Mewtwo-vs-Gligar where the agent had no scalar matchup-quality cue), weather
#   (Sunny Day / Rain / Sandstorm / Hail meaningfully shift damage and STAB), and
#   side conditions (Spikes / Reflect / Light Screen — switch-in cost and offensive pacing).
# Added (Agent16): per-slot rich features (own + opp), opp_seen_flags, own/opp_is_active,
#   force_switch_flag, speed_advantage_flag. Replaces the single species-scalar identity
#   that capped earlier agents at ~65/55% WR — the policy now has full type+base-stat
#   info per bench slot, can distinguish unseen vs seen-and-alive opp mons, and gets
#   explicit "active is here / forced to switch / I outspeed" signals.
# Added (Agent21): moves_type_onehot — explicit 18-dim type for each of 4 own moves.
#   Earlier agents only saw moves_dmg_multiplier (effectiveness vs CURRENT opp). They
#   could not reason "if I switch to a Water bench mon, does this move STILL hit hard?"
#   because move type was never visible — only its effectiveness vs the current matchup.
#   72 extra dims close that forward-reasoning gap (4 × 18). Appended at the END of the
#   obs so OWN_SLOT_OFFSET / OPP_SLOT_OFFSET are unchanged and the slot extractor
#   continues to slice the existing blocks correctly.
# Added (Agent23): obs space corrections + Gen 2 visibility upgrade.
#   BUG FIXES (no dim change):
#     - moves_dmg_multiplier default 1.0 -> 0.0 (was leaking "super-effective" for empty
#       slots, no-opp scenarios, and silent-asserted moves)
#     - moves_pp_ratio default 0.0 -> -1.0 (empty slots looked like 50% PP)
#     - moves_base_power None -> -1.0 (status moves looked like 125 BP attackers)
#   NEW INFO (+10 dims):
#     - Safeguard + Mist added to side_conditions (3 -> 5 dims each side, +4)
#       Both invisible to prior agents; Safeguard counters status leads, Mist counters
#       stat-drop strats — common in Gen 2 randombattle.
#     - sleep_counter (own + opp): turns slept / 7. Sleep lasts 1-7 turns; "wakes next
#       turn" vs "5 more turns asleep" was indistinguishable to prior agents.
#     - toxic_counter (own + opp): toxic stack / 15. Damage escalates 1/16, 2/16, ...;
#       a 1-stack vs 8-stack target is a totally different game.
#     - substitute (own + opp): 1.0 when behind a sub. Sub blocks status, stat changes,
#       and absorbs HP — bot was completely blind to it (Sub-CM Suicune etc.).
#   Side-condition expansion shifts OWN_SLOT_OFFSET 188 -> 192 and
#   OPP_SLOT_OFFSET 440 -> 444. The 6 new active-mon dims are appended at the tail
#   so the extractor's `tail = obs[:, OPP_SLOT_OFFSET + SLOT_BLOCK_LEN:]` slice
#   automatically picks them up.
OBS_SIZE = 794

ACTION_SPACE_SIZE = 11

# -----------------------------
# Slot-equivariant feature extractor (Agent19+)
# -----------------------------
# Agent16 added 252-dim own_slot_features and 252-dim opp_slot_features (6 slots
# × 42 dims = type1_onehot 18 + type2_onehot 18 + base_stats 6). Flat MLPs treat
# those 12 slot blocks as 504 independent features and have to re-learn the same
# Pokemon-encoding 12 separate times. A shared per-slot MLP (slot-equivariant)
# encodes once and broadcasts, reusing parameters across slots and across the
# own/opp banks. Empirically this is the standard fix for "value learns but
# policy stalls" on slot-structured obs (Agent16-18 symptom).
USE_SLOT_EQUIVARIANT = True

# Slot block layout inside the obs (must match embed_battle_impl).
# Agent23 expands side_conditions 3->5 each side (+4 dims in head), shifting both
# offsets by +4. New active-mon counters/sub flag (+6 dims) live at the TAIL after
# moves_type_onehot, so they don't move the slot blocks. Extractor's non_slot_dim
# is computed dynamically as OBS_SIZE - 2*SLOT_BLOCK_LEN, so the head/tail split adapts.
N_SLOTS = 6
SLOT_DIM = 42                  # 18 type1 + 18 type2 + 6 base_stats
OWN_SLOT_OFFSET = 192          # start of own_slot_features (Agent23: 188 + 4 from expanded side_conditions)
OPP_SLOT_OFFSET = 444          # start of opp_slot_features (= 192 + 252)
SLOT_BLOCK_LEN = N_SLOTS * SLOT_DIM   # 252
SLOT_HIDDEN = 32               # per-slot embedding size out of the shared MLP

# Separate pi/vf networks — prevents the policy and value function from
# competing over a shared 128-dim bottleneck, which caused value_loss to
# stay stuck at ~0.49 (vs ~0.16 for agents with smaller obs spaces).
# Each network sees the full features-extractor output and learns its own
# representation downstream of it. With slot-equivariant extractor the input
# dim is OBS_SIZE - 504 + 2*6*32 = 280 + 384 = 664 (Agent21, was 592 in Agent20
# with OBS_SIZE=712).
# Note: vf_coef only scales the value-net gradient; with separate networks it
# does NOT touch policy params, so vf_coef=1.0 is safe to combine with this arch.
# Agent17: wider pi (512→1024 first layer) — Agent16's value fn was excellent
# (vl=0.10, ev=0.55) but policy stuck at high entropy (-0.756) and negative
# rewards. [512,256,128] can't compress 712 dims into useful action probs.
# VF stays [512,256,128] since it doesn't need more capacity.
NET_ARCH = {"pi": [1024, 512, 256], "vf": [512, 256, 128]}

# FIX: Electrode's base Speed in Gen 2 is 150
MAX_SPEED = 150

LEARNING_RATE = 5e-5

# --- Parallel training config ---
# N_ENVS is the default; Main.py passes N_ENVS_RUN (currently 8) explicitly.
N_ENVS = 4
# With 8 envs at runtime: 4096 × 8 = 32,768 samples per rollout — better advantage
# estimation for ~40-turn episodes than the prior 2048×4 = 8192 setup.
N_STEPS = 4096
BATCH_SIZE = 512
N_EPOCHS = 8
# Agent22: 0.005 → 0.01. Agent21 entropy collapsed to -0.75 while losing
# (-0.42 reward). With ENT_COEF=0.005 there was almost no pressure to keep
# exploring — the policy locked into a deterministic losing strategy. Restoring
# 0.01 adds enough bonus to prevent premature commitment.
# Agent25: 0.01 → 0.012. Slight exploitation push paired with new reward-shaping
# signals (immunity penalty, flat voluntary-switch penalty). A24 entropy_loss
# sat at -0.72 (still very exploratory); the new shaping gives the policy more
# concrete gradient signal, so a small reduction in exploration pressure should
# help it commit. Staying well above A18's 0.005 (premature lock-in).
ENT_COEF = 0.012
# Raised from 0.5 — across Agent10/11 the value function lagged the policy
# (value_loss ~0.45, explained_variance trending DOWN toward 0.38 at 3.5M steps
# even with separated pi/vf nets). Doubling vf_coef gives the value head twice
# the gradient weight in the combined loss without touching the policy update.
VF_COEF = 1.0
# Agent21: 0.15 → 0.2 (revert toward Agent14's value).
# Agent20 ran clip_fraction at 0.22+ against the 0.15 cap — the policy was being
# clipped on >1 in 5 samples, every rollout. With explained_variance at 0.57 (the
# best of any agent) and policy_gradient_loss at -0.032 (also strongest), the
# critic + gradient direction were correct; the trust region was just too tight to
# act on them. Agent14 (clip=0.2) was the best historical performer; every agent
# since at 0.15 has done worse. Reverting to 0.2 with the new equivariant arch.
# Prior context (Agent17 collapse): a single rollout cratered ep_rew_mean to -0.877
# at 2.4M steps with wider pi + clip 0.2. Agent19+ adds slot-equivariant extractor
# which makes per-slot reasoning much cheaper, reducing the chance of one-shot
# blowups in the value head when policy updates loosen.
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.92
# Agent22: 0.03 → 0.02. Agent21 was hitting approx_kl=0.032–0.038, barely
# triggering early-stop despite the 0.03 cap — the constraint was too loose to
# prevent policy thrashing. Agent14's healthy operating range was ~0.013; 0.02
# matches that without over-throttling the wider pi network.
TARGET_KL = 0.02
# Agent19: 0.99 → 0.995. Effective horizon goes from ~100 steps to ~200 steps.
# Pokemon battles last ~40 turns and setup-then-sweep strategies (Curse,
# Swords Dance, Belly Drum) take 3-5 turns to pay off. The Agent14-18 pattern
# of "value head learns but policy can't act" is consistent with too-short
# credit assignment: by the time a Curse turn pays off as a KO, the discounted
# return at the Curse step is heavily attenuated. Lengthening the horizon lets
# the policy connect setup → KO directly.
GAMMA = 0.995

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
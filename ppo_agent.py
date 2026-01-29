import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.player.baselines import RandomPlayer
from poke_env.data import GenData

# Pokémon type list (stable)

POKEMON_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic",
    "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
]


def type_one_hot(pokemon):
    vec = [0.0] * len(POKEMON_TYPES)
    if pokemon:
        for t in pokemon.types:
            if t in POKEMON_TYPES:
                vec[POKEMON_TYPES.index(t)] = 1.0
    return vec


# ----------------------------
# Policy Network
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# PPO Agent 
# ----------------------------
class PPOAgent(Player):
    def __init__(self, battle_format="gen1randombattle", log_level=30):
        super().__init__(battle_format=battle_format, log_level=log_level)

        # Updated input size: 2 (HP) + 18 (my types) + 18 (opp types) + 12 (4 moves × 3 features)
        self.input_size = 50
        # Output size: 4 moves + 5 switches = 9 total actions
        self.policy = PolicyNetwork(self.input_size, 128, 9)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.last_log_prob = None
        self._last_moves = []
        self._switch_count = 0  # Track consecutive switches
        self._last_action_was_switch = False
        
        # Get type chart for Gen 1
        self.gen_data = GenData.from_gen(1)

    # ----------------------------
    # State representation
    # ----------------------------
    def battle_to_tensor(self, battle):
        features = []

        # ---- HP ----
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0
        features += [my_hp, opp_hp]

        # ---- My Pokémon types (fixed - only once) ----
        features += type_one_hot(battle.active_pokemon)

        # ---- Opponent Pokémon types (fixed - only once) ----
        features += type_one_hot(battle.opponent_active_pokemon)

        # ---- Move features (up to 4) ----
        for i in range(4):
            if i < len(battle.available_moves):
                move = battle.available_moves[i]

                bp = move.base_power / 100 if move.base_power else 0.0

                if battle.opponent_active_pokemon:
                    # Convert types list to tuple for damage_multiplier
                    eff = move.type.damage_multiplier(
                        *battle.opponent_active_pokemon.types,
                        type_chart=self.gen_data.type_chart
                    )
                else:
                    eff = 1.0

                stab = 1.0 if (
                    battle.active_pokemon
                    and move.type in battle.active_pokemon.types
                ) else 0.0

                features += [bp, eff, stab]
            else:
                features += [0.0, 0.0, 0.0]

        return torch.tensor(features, dtype=torch.float32)

    # ----------------------------
    # Choose action
    # ----------------------------
    def choose_move(self, battle):
        try:
            # Force attack if we've switched already (removes repetitive or unneeded switching)
            force_attack = self._switch_count >= 1
            
            x = self.battle_to_tensor(battle)
            logits = self.policy(x)
            
            # BIAS TOWARD ATTACKING: Add +2.0 to move logits to make them more attractive
            logits[:4] += 2.0
            
            # If forcing attack, mask out switch actions
            if force_attack:
                logits[4:] = float('-inf')  # Make switch actions impossible
            
            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            self.last_log_prob = dist.log_prob(action_idx)

            # Actions 0-3: Moves
            # Actions 4-5: Switches (5 possible switches since 6 total - 1 active)
            
            if action_idx < 4:
                # Choose a move
                available = [
                    m for m in battle.available_moves
                    if m.id not in ("metronome", "assist", "transform", "copycat", "mirrormove", "mimic", "mefirst")
                ]

                if not available:
                    return self.choose_random_move(battle)

                move = available[action_idx.item() % len(available)]
                self._last_moves.append(move)
                
                # Reset switch counter when attacking
                self._switch_count = 0
                self._last_action_was_switch = False
                
                return self.create_order(move)
            
            else:
                # Choose a switch
                available_switches = [
                    p for p in battle.available_switches
                ]
                
                if not available_switches or force_attack:
                    # If no switches available or forced to attack, pick a move instead
                    available = [
                        m for m in battle.available_moves
                        if m.id not in ("metronome", "assist", "transform", "copycat", "mirrormove", "mimic", "mefirst")
                    ]
                    if available:
                        move = available[0]
                        self._last_moves.append(move)
                        self._switch_count = 0
                        self._last_action_was_switch = False
                        return self.create_order(move)
                    return self.choose_random_move(battle)
                
                # Map action index 4-8 to switch index 0-4
                switch_idx = (action_idx.item() - 4) % len(available_switches)
                pokemon = available_switches[switch_idx]
                
                # Increment switch counter
                self._switch_count += 1
                self._last_action_was_switch = True
                
                return self.create_order(pokemon)

        except Exception as e:
            print("Move error:", e)
            return self.choose_random_move(battle)

    # ----------------------------
    # Reward + update
    # ----------------------------
    async def on_battle_end(self, battle, won):
        # Penalty for ties
        if battle.won is None:  # Tie
            reward = -0.5
        else:
            reward = 1.0 if won else -1.0

        # HP differential bonus
        my_hp = sum(p.current_hp_fraction for p in battle.team.values())
        opp_hp = sum(p.current_hp_fraction for p in battle.opponent_team.values())
        reward += 0.25 * (my_hp - opp_hp)

        # Penalty for own fainted Pokémon
        fainted_own = sum(1 for p in battle.team.values() if p.current_hp == 0)
        reward -= 0.05 * fainted_own

        # REWARD for enemy fainted Pokémon
        fainted_enemy = sum(1 for p in battle.opponent_team.values() if p.current_hp == 0)
        reward += 0.06 * fainted_enemy

        # Type effectiveness bonuses
        for move in self._last_moves:
            if battle.opponent_active_pokemon:
                # Convert types list to tuple for damage_multiplier
                eff = move.type.damage_multiplier(
                    *battle.opponent_active_pokemon.types,
                    type_chart=self.gen_data.type_chart
                )
                if eff > 1:
                    reward += 0.02
                elif eff < 1:
                    reward -= 0.01

            if battle.active_pokemon and move.type in battle.active_pokemon.types:
                reward += 0.02

        if self.last_log_prob is not None:
            loss = -self.last_log_prob * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.last_log_prob = None
        self._last_moves = []
        self._switch_count = 0
        self._last_action_was_switch = False

        print(
            f"Battle ended | Won: {won} | Reward: {reward:.2f} | "
            f"My HP: {my_hp:.2f} | Opp HP: {opp_hp:.2f} | "
            f"My fainted: {fainted_own} | Enemy fainted: {fainted_enemy}"
        )


# ----------------------------
# Training loop
# ----------------------------
async def train():
    agent = PPOAgent()
    opponent = SimpleHeuristicsPlayer(battle_format="gen1randombattle")

    n_battles = 2000
    print("Starting training...\n")

    for i in range(n_battles):
        print(f"Battle {i + 1}/{n_battles}")
        await agent.battle_against(opponent, n_battles=1)

    print("\nTraining complete!")


if __name__ == "__main__":
    asyncio.run(train())
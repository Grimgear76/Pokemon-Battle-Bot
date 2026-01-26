import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer

# ---- Neural Network ----
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# ---- PPO Agent ----
class PPOAgent(Player):
    def __init__(self, battle_format="gen9randombattle", log_level=30):
        super().__init__(battle_format=battle_format, log_level=log_level)
        self.policy = PolicyNetwork(input_size=10, hidden_size=64, output_size=4)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.last_log_prob = None

        # Track moves manually
        self._last_moves = []

    def choose_move(self, battle):
        try:
            # Create input tensor
            x = self.battle_to_tensor(battle)
            logits = self.policy(x)
            m = Categorical(logits=logits)
            action_idx = m.sample()
            self.last_log_prob = m.log_prob(action_idx)

            # Pick move safely
            available = [m for m in battle.available_moves if m.id not in ("mirrormove", "metronome", "assist", "transform", "copycat")]
            if available:
                move = available[action_idx.item() % len(available)]
                # Track move for reward
                self._last_moves.append({
                    "move": move.id,
                    "damage_done": getattr(move, "damage_done", 0),
                    "damage_taken": getattr(move, "damage_taken", 0),
                    "super_effective": getattr(move, "super_effective", False)
                })
                return self.create_order(move)
            else:
                return self.choose_random_move(battle)

        except Exception as e:
            # Skip problematic moves
            print("Skipping unhandled move:", e)
            return self.choose_random_move(battle)

    def battle_to_tensor(self, battle):
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0
        status = 0
        return torch.tensor([my_hp, opp_hp, status] + [0]*7, dtype=torch.float32)

    async def on_battle_end(self, battle, won):
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0

        # Base reward
        reward = 1.0 if won else -1.0
        reward += 0.5 * (my_hp - opp_hp)

        # Heuristic bonuses
        for move in self._last_moves:
            if move.get("super_effective", False):
                reward += 0.03
            reward += 0.02 * move.get("damage_done", 0)
            reward -= 0.01 * move.get("damage_taken", 0)

        # Bonus for fainted enemy Pokémon
        fainted_enemies = sum(1 for p in battle.opponent_team.values() if p.current_hp == 0)
        reward += 0.05 * fainted_enemies

        # Update policy
        if self.last_log_prob is not None:
            loss = -self.last_log_prob * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear moves for next battle
        self._last_moves = []

        # Print outcome
        print(f"Battle finished | Won: {won} | Reward: {reward:.2f} | My HP: {my_hp:.2f} | Opponent HP: {opp_hp:.2f} | Fainted enemies: {fainted_enemies}")


# ---- Training function ----
async def train():
    agent = PPOAgent()
    opponent = SimpleHeuristicsPlayer(battle_format="gen9randombattle")  # safe heuristic agent
    n_battles = 500

    print("Starting training...\n")
    for i in range(n_battles):
        print(f"Starting battle {i+1}...")
        await agent.battle_against(opponent, n_battles=1)

    print("\nTraining finished!")


# ---- Run ----
if __name__ == "__main__":
    asyncio.run(train())

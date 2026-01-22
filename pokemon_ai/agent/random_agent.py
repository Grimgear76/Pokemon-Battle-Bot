import random
from poke_env.player import Player

class RandomAgent(Player):
    """
    Random move agent for Pokémon Showdown.
    Works with latest poke-env (connects to the public server automatically).
    """

    def choose_move(self, battle):
        if battle.available_moves:
            return random.choice(battle.available_moves)
        return self.choose_random_move(battle)

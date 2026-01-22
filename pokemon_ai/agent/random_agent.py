import random
from poke_env.player import Player

class RandomAgent(Player):
    """
    Baseline agent that picks a random move.
    Connects to the public Pokémon Showdown server.
    """

    def __init__(self, battle_format):
        super().__init__(
            battle_format=battle_format,
            server="sim.smogon.com",  # public server
            port=443,                  # HTTPS port
            use_ssl=True               
        )

    def choose_move(self, battle):
        if battle.available_moves:
            return random.choice(battle.available_moves)
        return self.choose_random_move(battle)

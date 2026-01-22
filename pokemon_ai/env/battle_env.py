class BattleEnv:
    def __init__(self, agent_cls, opponent_cls, battle_format):
        self.agent = agent_cls(battle_format=battle_format)
        self.opponent = opponent_cls(battle_format=battle_format)

    async def run(self, n_battles=1):
        await self.agent.battle_against(self.opponent, n_battles=n_battles)

    def get_results(self):
        return {
            "wins": self.agent.n_won_battles,
            "losses": self.agent.n_lost_battles,
        }

from dataclasses import dataclass


@dataclass
class StrategyResults:
    reward: float
    day_fees: float
    net_reward: float
    avg_pips_risked: float
    n_buys: int
    n_sells: int
    n_wins: int
    n_losses: int
    longest_win_streak: int
    longest_loss_streak: int

    def __str__(self):
        return f'RESULTS:\nreward = {self.reward}\nday fees = {self.day_fees}\nnet reward = {self.net_reward}' \
               f'\navg pips risked = {self.avg_pips_risked}\nbuys = {self.n_buys}\nsells = {self.n_sells}' \
               f'\nwins = {self.n_wins}\nlosses = {self.n_losses}\nlongest win streak = {self.longest_win_streak}' \
               f'\nlongest loss streak = {self.longest_loss_streak}'

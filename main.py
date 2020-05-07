import matplotlib.pyplot as plt
import os
import seaborn as sns

from pandas import DataFrame
from tic_tac_toe.game import Player, Game
from tic_tac_toe.agents.console_input_agent import ConsoleInputAgent
from tic_tac_toe.agents.random_agent import RandomAgent
from tic_tac_toe.agents.minimax_agent import MinimaxAgent
from tic_tac_toe.agents.timed_agent import TimedAgent
from tic_tac_toe.agents.alpha_beta_agent import AlphaBetaAgent

sns.set(style="whitegrid")

AGENTS = [
    ("Human", ConsoleInputAgent),
    ("Random Agent", RandomAgent),
    ("Minimax Agent", MinimaxAgent),
    ("Alpha-Beta Agent", AlphaBetaAgent)
]

def _pick_agent(player, board_size = 3, stats=None):
    def _try_pick():
        try:
            list_of_agents = "\n".join(
                map(lambda x: "\t{} - {}".format(x[0], x[1][0]),
                    enumerate(AGENTS)))
            agent = int(
                input("Available agents: \n{}\nPick an agent [0-{}]: ".format(
                    list_of_agents, len(AGENTS) - 1)))
            return agent
        except ValueError:
            return None

    agent = _try_pick()

    while agent is None:
        print("Incorrect selection, try again.")
        agent = _try_pick()

    return TimedAgent(AGENTS[agent][1](player), board_size, stats=stats)


def main():
    stats = {'turn':[], 'runtime':[], 'states_visited':[]}

    print("Choosing player X...")
    player_x = _pick_agent(Player.X, stats=stats)

    print("Choosing player O...")
    player_o = _pick_agent(Player.O, stats=stats)
    play = "y"

    wins = [0] * 3
    while play == "y":
        game = Game(player_x, player_o)
        wins[game.play()] += 1
        print("x: {} | o: {} | {} draws".format(wins[Player.X],wins[Player.O],wins[2]))
        play = input("Play again? y/[n]: ")

    if input("Save data? y/[n]: ") == "y":
        folder = "plots/" + input("folder: plots/")
        os.mkdir(folder)

        ax = sns.violinplot(x="turn", y="runtime", data=stats)
        ax.set(xlabel='turn', ylabel='runtime (seconds)')
        plt.tight_layout()
        plt.savefig(folder + "/runtime")

        plt.cla()
        ax = sns.violinplot(x="turn", y="states_visited", data=stats)
        ax.set(xlabel='turn', ylabel='states visited')
        plt.savefig(folder + "/states_visited")

        stats_file = open(folder + '/stats', 'w')
        df = DataFrame(stats)
        means = df.groupby(['turn']).mean()
        for i in range(len(means)):
            stats_file.write("(%d,%f)\n"%(i+1, means['runtime'].loc[i+1].item()))
        stats_file.write("\n")
        try:
            for i in range(len(means)):
                stats_file.write("(%d,%d)\n"%(i+1, means['states_visited'].loc[i+1].item()))
        except:
            pass
        stats_file.close()

if __name__ == "__main__":
    main()

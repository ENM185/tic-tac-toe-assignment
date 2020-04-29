from tic_tac_toe.game import Player, Game
from tic_tac_toe.agents.console_input_agent import ConsoleInputAgent
from tic_tac_toe.agents.random_agent import RandomAgent
from tic_tac_toe.agents.minimax_agent import MinimaxAgent

AGENTS = [
    ("Human", ConsoleInputAgent),
    ("Random Agent", RandomAgent),
    ("Minimax Agent", MinimaxAgent)
]


def _pick_agent(player):
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

    return AGENTS[agent][1](player)


def main():
    print("Choosing player X...")
    player_x = _pick_agent(Player.X)

    print("Choosing player O...")
    player_o = _pick_agent(Player.O)
    play = "y"

    while play == "y":
        game = Game(player_x, player_o)
        game.play()
        play = input("Play again? y/[n]: ")


if __name__ == "__main__":
    main()

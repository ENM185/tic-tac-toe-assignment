class Player:
    X = 0
    O = 1

    ALL_PLAYERS = [X, O]


PLAYER_NAMES = {
    Player.X: "x",
    Player.O: "o",
}


def other_player(player):
    if player == Player.X:
        return Player.O
    elif player == Player.O:
        return Player.X

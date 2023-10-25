import numpy as np

from tiktactoe import tiktactoe as tt

from MCTS import MCTS

game = tt()
args = {"ep" : 1.41, "n_searches" : 1000}
opp = MCTS(game, args)


player = 1
state = game.get_init_state()
while True:
    if player == 1:
        move = int(input())
    else:
        flipped = game.change_prespective(state, player)
        actions = opp.search(flipped)
        print(actions )
        move = np.argmax(actions)
    state = game.get_next_state(state,move,player)
    print(state.reshape(game.nrows,game.nrows))
    value,teminal = game.get_value_and_terminate(state,move)
    print(value,teminal,move)
    if teminal:
        if player == 1:
            print("player won!")
        else:
            print("PC won!")
        break
    player = game.get_apponent(player)
    print("---------------------------\n\n")
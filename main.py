import numpy as np
import torch

from tiktactoe import tiktactoe as tt

from MCTS import MCTS
from alphago import alphazero

game = tt()
args = {"ep": 2, "n_searches": 1000, "n_iterations": 3, "nbr_examples": 500, "nbr_epochs": 4}
mcts = MCTS
opp = alphazero(game, mcts, args)

checkpoint = "checkpoints/model_0.pth"


if checkpoint == "":
    opp.learn()
else:
    print("loading model....")
    opp.model.load_state_dict(torch.load(checkpoint))
    print("model loaded!")

state = game.get_init_state()
teminal = False
player = -1
while not teminal:
    game.display(state)
    if player == 1:
        action = int(input("Select a move "))#opp.inferance(state,player)
    else:
        action = opp.inferance(state,player)
    state = game.get_next_state(state, action, player)
    value, teminal = game.get_value_and_terminate(state, action)


    player = game.get_apponent(player)

game.display(state)
if value == 1:
    if player == -1:
        print("player won")
    else:
        print("alpha won")
else:
    print("draw :(")



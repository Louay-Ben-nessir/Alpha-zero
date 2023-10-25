import numpy as np
#this itself is not a state it's more so functions to cntrl the env
class tiktactoe:
    def __init__(self, ncols = 3, nrows = 3):
        self.ncols = ncols
        self.nrows = nrows
        self.n_actions = self.ncols * self.nrows

    def get_init_state(self):
        return np.zeros(self.n_actions)

    def get_next_state(self,state , action, player):
        state[action] = player
        return state

    def get_valid_moves(self, state):
        return (state == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None:
            return False

        row = action // self.ncols
        column = action % self.nrows
        player = state[action]
        state = state.reshape((self.nrows, self.ncols))

        return (
                np.sum(state[row, :]) == player * self.ncols
                or np.sum(state[:, column]) == player * self.nrows
                or np.sum(np.diag(state)) == player * self.nrows
                or np.sum(np.diag(np.flip(state, axis=0))) == player * self.nrows
        )

    def get_value_and_terminate(self,state, action):
        if self.check_win(state, action):
            return 1, True
        elif self.get_valid_moves(state).sum() == 0:
            return 0, True
        return 0, False

    def get_apponent(self, player):
        return -player

    def get_apponent_value(self, value):
        return -value

    def change_prespective(self,state,player):
        return state * player

    def get_encoded_state(self, state):
        state = state.reshape((self.nrows, self.ncols))
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        return encoded_state

    def display(self,state):
        state = state.reshape(self.nrows, self.nrows)
        block = "{} | {} | {}"
        line = "-" * len(block)
        for idx,row in enumerate(state):
            nrow = []
            for idx2,elem in enumerate(row):
                match elem:
                    case 1:
                        nrow.append("x")
                    case -1:
                        nrow.append("o")
                    case _:
                        nrow.append(idx * 3 + idx2)
            print(block.format(*nrow))
            if idx != len(state) - 1:
                print(line)
        print("")






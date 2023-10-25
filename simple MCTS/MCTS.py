import numpy as np
import math


class Node:
    def __init__(self, game, args, state, parent=None, action=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []
        self.possible_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.win_count = 0

    def is_fully_expanded(self):
        return not any(self.possible_moves) and len(self.children) > 0

    def select(self):
        return max(((self.calc_ucb(child), child) for child in self.children), key=lambda x: x[0])[1]

    def calc_ucb(self, child):
        q_val = 1 - ((child.win_count / child.visit_count) + 1) / 2
        return q_val + self.args["ep"] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        action = np.random.choice(np.where(self.possible_moves == 1)[0])
        self.possible_moves[action] = 0

        child_state = self.game.get_next_state(self.state.copy(), action, 1)
        child_state = self.game.change_prespective(child_state, -1)

        child = Node(self.game, self.args, child_state, self, action)

        self.children.append(child)

        return child

    def simulate(self):
        value, terminal = self.game.get_value_and_terminate(self.state, self.action)
        value = self.game.get_apponent_value(value)
        if terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:

            action = np.random.choice(np.where(self.game.get_valid_moves(rollout_state) == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, terminal = self.game.get_value_and_terminate(rollout_state, action)

            if terminal:
                if rollout_player == -1:
                    return self.game.get_apponent_value(value)
                return value
            rollout_player = self.game.get_apponent(rollout_player)

    def backpropagate(self, value):
        self.win_count += value
        self.visit_count += 1

        if self.parent is not None:
            value = self.game.get_apponent_value(value)
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args: dict):
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state)
        for _ in range(self.args["n_searches"]):
            node = root
            while node.is_fully_expanded():  # continue untill we find a node that we can expand
                node = node.select()
            value, terminal = self.game.get_value_and_terminate(node.state, node.action)
            value = self.game.get_apponent_value(value)
            if not terminal:
                node = node.expand()
                value = node.simulate()
            node.backpropagate(value)

        actions = np.zeros(self.game.n_actions)
        for child in root.children:
            actions[child.action] = child.visit_count
        actions /= actions.sum()
        return actions

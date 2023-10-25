import numpy as np
import math

import torch


class Node:
    def __init__(self, game, args, state, parent=None, action=None, prior = 0, visit_count = 0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.win_count = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        return max(((self.calc_ucb(child), child) for child in self.children), key=lambda x: x[0])[1]

    def calc_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.win_count / child.visit_count) + 1) / 2
        return q_value + self.args['ep'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_prespective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)



    def backpropagate(self, value):
        self.win_count += value
        self.visit_count += 1

        if self.parent is not None:
            value = self.game.get_apponent_value(value)
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, model, args: dict):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['d_ep']) * policy + self.args['d_ep'] \
                 * np.random.dirichlet([self.args['d_al']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for _ in range(self.args["n_searches"]):
            node = root
            while node.is_fully_expanded():  # continue untill we find a node that we can expand
                node = node.select()
            value, terminal = self.game.get_value_and_terminate(node.state, node.action)
            value = self.game.get_apponent_value(value)
            if not terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=-1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves #filter out unplayable moves
                policy /= np.sum(policy) # all values sum up to 1

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.n_actions)
        for child in root.children:
            action_probs[child.action] = child.visit_count
        action_probs /= np.sum(action_probs)

        return action_probs

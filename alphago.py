import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm, trange
import timm
import numpy as np

class alphann(nn.Module):
    def __init__(self,game, model_name="resnet18", input_size=3):
        super().__init__()
        self.base = timm.create_model(model_name,num_classes = 0)

        num_hidden = self.base.num_features
        self.policyHead = nn.Sequential(
            nn.Linear(num_hidden, game.n_actions)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.base(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class MemeoryDataset(Dataset):
    def __init__(self, mem):
        self.mem = mem

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, index):
        state, action_probs,value = (torch.tensor(elem, dtype=torch.float32) for elem in self.mem[index])
        return  state, action_probs,value


class alphazero:
    def __init__(self, game, MCTS, args, device):
        self.model = alphann(game)
        self.model.eval()
        self.model.to(device)
        self.game = game
        self.args = args
        self.mcts = MCTS(game, self.model, args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        self.device = device


    def selfplay(self):
        memory = []
        player = 1
        state = self.game.get_init_state()
        while True:
            neutral_state = self.game.change_prespective(state, player)
            action_probs = self.mcts.search(neutral_state)
            memory.append([self.game.get_encoded_state(neutral_state), action_probs, player])

            taction_probs = action_probs ** (1 / self.args['temp'])
            action = np.random.choice(self.game.n_actions, p=taction_probs)

            state = self.game.get_next_state(state, action, player)
            value, teminal = self.game.get_value_and_terminate(state,action)

            if teminal:
                return map(lambda x: (x[0], x[1], value if x[2] == player else self.game.get_apponent_value(value)), memory)
            player = self.game.get_apponent(player)
    def train(self, data_loader):
        avg_loss = 0
        for state, policy,value in tqdm(data_loader, total = len(data_loader)):
            state = state.to(self.model.device)
            policy_pred, value_pred = self.model(state)
            loss = self.policy_loss(policy_pred,policy) + self.value_loss(value_pred.squeeze(1),value)
            avg_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("loss ",avg_loss /len(data_loader))

    def learn(self):
        for iteration in trange(self.args["n_iterations"]):
            self.model.eval()
            memory = []
            for _ in range(self.args["nbr_examples"]):
                memory.extend(self.selfplay())

            self.model.train()
            memory = MemeoryDataset(memory)
            for epoch in range(self.args["nbr_epochs"]):
                data_loader = DataLoader(memory, batch_size=64, shuffle=True)
                self.train(data_loader)
            torch.save(self.model.state_dict(),f"checkpoints/model_{iteration}.pth")


    def inferance(self,state,player):
        #assuming the opponante is allways -1
        neutral_state = self.game.change_prespective(state, player)
        mcts_probs = self.mcts.search(neutral_state)
        return np.argmax(mcts_probs)
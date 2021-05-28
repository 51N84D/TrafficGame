from models import MLP
import torch
from pathlib import Path


class Agent:
    def __init__(self, position, name, agent_type, input_dim, lr):
        super().__init__()
        self.position = position  # (x, y) in grid world
        self.type = agent_type  # horizontal or vertical
        self.name = name
        self.input_dim = input_dim
        self.lr = lr

        self.policy_network = None
        self.optimizer = None
        self.initialize_policy_network()
        self.get_optimizer()
        self.done = False
        self.collided = False

    def get_action(self, noise):
        if self.type == "h":
            input_aug = torch.ones(noise.shape)
        elif self.type == "v":
            input_aug = torch.zeros(noise.shape)
        else:
            raise ValueError(f"Agent type {self.type} is invalid")

        policy_input = torch.cat((noise, input_aug))
        stoch_action = self.policy_network(policy_input)
        action = torch.argmax(stoch_action).numpy()
        return stoch_action, action

    def initialize_policy_network(self):
        # Add 1 to input dim to account for agent type
        self.policy_network = MLP(
            input_size=self.input_dim * 2,
            hidden_size=16,
            n_hidden_layers=2,
            num_classes=2,
        )

    def get_optimizer(self):
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def save_weights(self, save_dir):
        save_dir = Path(save_dir).resolve()
        torch.save(
            self.policy_network.state_dict(), save_dir / f"agent_{self.name}.pth"
        )

    def load_weights(self, load_dir):
        load_dir = Path(load_dir).resolve()
        self.policy_network.load_state_dict(
            torch.load(load_dir / f"agent_{self.name}.pth")
        )

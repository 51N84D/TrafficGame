from pathlib import Path
from agent import Agent
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
import pickle


class GridGame:
    def __init__(
        self,
        grid_dim,
        noise_dim,
        save_dir,
        n_train_iter,
        n_save_iter,
        n_plot_iter=100,
        load_dir=None,
        resume=False,
        comet_exp=None,
        lr=1e-3,
    ):
        super().__init__()

        self.lr = lr

        if load_dir is None:
            self.load_dir = Path(save_dir)
        else:
            self.load_dir = Path(load_dir)
        self.save_dir = Path(save_dir).resolve()

        if resume:
            print("Loading saved environment settings...")
            self.load_env_details()
            self.agents = self.initialize_agents()
            self.load_agent_weights()

        else:
            self.grid_dim = grid_dim
            self.noise_dim = noise_dim
            self.n_agents = (grid_dim - 2) * 2
            self.agents = self.initialize_agents()

        self.n_train_iter = n_train_iter
        self.n_save_iter = n_save_iter
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.grid = np.zeros((self.grid_dim, self.grid_dim))
        self.set_grid()
        self.no_collision_payoff = torch.from_numpy(np.asarray([0, 1])).float()
        self.collision_payoff = torch.from_numpy(
            np.asarray(
                [
                    [0, 0],
                    [1, -50],
                ]
            )
        ).float()
        self.current_payoffs = []  # List of lists, length = number of steps
        self.average_payoffs = []  # List of lists, length = number of steps
        self.n_plot_iter = n_plot_iter

    def initialize_agents(self):
        agents = []
        for i in range(self.n_agents):
            if i < int(self.n_agents / 2):
                agent_type = "h"
                agent_pos = (0, (i + 1))
            else:
                agent_type = "v"
                agent_pos = (int(i % (self.n_agents / 2)) + 1, 0)

            agent = Agent(
                position=agent_pos,
                name=str(i),
                agent_type=agent_type,
                input_dim=self.noise_dim,
                lr=self.lr,
            )  # Agent that travels horizontally in first step
            agents.append(agent)
        return agents

    def reset_agent_positions(self):
        for i, agent in enumerate(self.agents):
            agent.done = False
            if i < int(self.n_agents / 2):
                agent_pos = (0, (i + 1))
            else:
                agent_pos = (int(i % (self.n_agents / 2)) + 1, 0)
            agent.position = agent_pos

    def clear_screen(self):
        os.system("clear")

    def render(self):
        self.clear_screen()
        print("------------------------------------------------------------")
        for y_idx, row in enumerate(self.grid):
            for x_idx, col in enumerate(row):
                if self.grid[x_idx, y_idx] > 0:
                    # Second pass to print stuff
                    if self.grid[x_idx, y_idx] == 2:
                        print("X", end="\t")
                    else:
                        for agent in self.agents:
                            if agent.position == (x_idx, y_idx):
                                print(agent.name, end="\t")
                else:
                    print("-", end="\t")
            print("\n")
        print("------------------------------------------------------------")

    def step(self, train=True):
        self.reset_grid()
        noise = self.sample_noise()
        agent_stoch_actions, agent_actions = self.get_agent_actions(noise)
        self.move_agent_positions(agent_actions)
        self.set_grid(agent_actions=agent_actions)
        agent_payoffs = self.get_payoffs(agent_actions, agent_stoch_actions)
        self.current_payoffs.append(agent_payoffs)
        if train:
            self.update_agent_weights(agent_payoffs)
        else:  # display payoffs
            # print("noise: ", noise)
            avg_payoff = np.nanmean(np.asarray([i.detach().numpy() if type(i) != float else i for i in agent_payoffs]))
            print("average payoff: ", avg_payoff)
        self.check_if_agents_are_done()

    def reset_grid(self):
        self.grid = np.zeros((self.grid_dim, self.grid_dim))

    def sample_noise(self):
        return torch.from_numpy(np.random.normal(0, 1, size=(self.noise_dim))).float()

    def get_agent_actions(self, noise):
        agent_actions = []
        agent_stoch_actions = []
        for i, agent in enumerate(self.agents):
            stoch_action, action = agent.get_action(noise)
            agent_stoch_actions.append(stoch_action)
            agent_actions.append(action)
        return agent_stoch_actions, agent_actions

    def check_if_agents_are_done(self):
        for i, agent in enumerate(self.agents):
            if (
                agent.position[0] == self.grid_dim - 1
                or agent.position[1] == self.grid_dim - 1
            ):
                agent.done = True

    def move_agent_positions(self, agent_actions):
        for i, agent in enumerate(self.agents):
            if agent.done:
                continue
            action = agent_actions[i]
            if action == 1:
                if agent.type == "v":  # vertical agent
                    agent.position = (agent.position[0], agent.position[1] + 1)
                else:  # horizontal agent
                    agent.position = (agent.position[0] + 1, agent.position[1])

    def get_payoffs(self, agent_actions, agent_stoch_actions):
        # for each agent, get the current payoff
        agent_payoffs = []
        for i, agent in enumerate(self.agents):
            if agent.done:
                agent_payoffs.append(np.nan)
                continue
            stoch_action = agent_stoch_actions[i]
            if self.grid[agent.position] == 1:  # no collision
                payoff = torch.matmul(stoch_action, self.no_collision_payoff)
            elif self.grid[agent.position] == 2:  # collision
                collided_idx = None
                for j, agent2 in enumerate(self.agents):
                    if agent2 == agent:
                        continue
                    if agent2.position == agent.position:
                        collided_idx = j
                        break

                # Only collision if both players pick "Go"
                if agent_actions[i] == 1 and agent_actions[j] == 1:
                    payoff = torch.matmul(
                        torch.matmul(stoch_action, self.collision_payoff),
                        agent_stoch_actions[collided_idx].detach(),
                    )
                else:
                    payoff = torch.matmul(stoch_action, self.no_collision_payoff)
            else:
                raise ValueError(f"Grid value {self.grid[agent.position]} is invalid")
            agent_payoffs.append(payoff)
        return agent_payoffs

    def update_agent_weights(self, agent_payoffs):
        for i, agent in enumerate(self.agents):
            if agent.done:
                continue
            payoff = agent_payoffs[i]
            loss = -payoff
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

    def set_grid(self, agent_actions=None):
        self.reset_grid()
        for agent in self.agents:
            x_idx, y_idx = agent.position
            if self.grid[x_idx, y_idx] == 1:
                self.grid[x_idx, y_idx] = 2
            else:
                self.grid[x_idx, y_idx] = 1

    def save_agent_weights(self):
        for i, agent in enumerate(self.agents):
            agent.save_weights(self.save_dir)

    def load_agent_weights(self):
        for i, agent in enumerate(self.agents):
            agent.load_weights(self.load_dir)

    def play_one_game(self, display=False, train=True):
        self.reset_agent_positions()
        if display:
            self.render()
            time.sleep(0.5)
        for i in range((self.grid_dim - 1) * 2):
            self.step(train=train)
            if display:
                self.render()
                time.sleep(0.5)

    def train(self):
        for i in tqdm(range(self.n_train_iter)):
            self.play_one_game()
            if i % self.n_save_iter == 0 and i != 0:
                self.save_agent_weights()
                self.save_env_details()
            if i % self.n_plot_iter == 0 and i != 0:
                self.average_payoffs.append(
                    np.nanmean(np.asarray(self.current_payoffs), axis=0)
                )
                self.current_payoffs = []
        self.plot_payoffs()

    def test(self):
        self.load_agent_weights()
        self.play_one_game(display=True, train=False)

    def plot_payoffs(self):
        self.average_payoffs = np.asarray(self.average_payoffs)
        color = iter(cm.rainbow(np.linspace(0, 1, self.n_agents)))
        for i in range(self.n_agents):
            c = next(color)
            plt.plot(
                np.arange(self.average_payoffs.shape[0]) * self.n_plot_iter,
                self.average_payoffs[:, i],
                c=c,
                markersize=0.1,
                linewidth=0.3,
                label=f"agent {i}",
            )

        plt.legend()

        plt.savefig(str(self.save_dir / "losses.png"))

    def load_env_details(self):
        with open(str(self.load_dir / "env_details.pkl"), "rb") as f:
            env_details = pickle.load(f)

        self.grid_dim = env_details["grid_dim"]
        self.n_agents = env_details["n_agents"]
        self.noise_dim = env_details["noise_dim"]

    def save_env_details(self):
        # Save details about the environment
        # self.n_agents = len(glob.glob1(self.load_dir, "*.pth"))
        # self.grid_dim =
        env_details = {
            "grid_dim": self.grid_dim,
            "n_agents": self.n_agents,
            "noise_dim": self.noise_dim,
        }
        with open(str(self.save_dir / "env_details.pkl"), "wb") as f:
            pickle.dump(env_details, f, pickle.HIGHEST_PROTOCOL)

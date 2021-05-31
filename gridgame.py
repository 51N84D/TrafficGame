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
from celluloid import Camera


class GridGame:
    def __init__(
        self,
        n_agents_one_side,
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
            self.grid_dim = (n_agents_one_side * 2) + 1
            self.n_agents = n_agents_one_side * 2
            self.noise_dim = noise_dim
            self.agents = self.initialize_agents()

        self.n_train_iter = n_train_iter
        self.n_save_iter = n_save_iter
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # NOTE: positions are decentralized (stored with each agent object)
        # Grid is used to keep track of number of agents at each position
        self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int8)

        # To keep track of collisions
        self.collision_grid = np.zeros((self.grid_dim, self.grid_dim))
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

        # Plotting variables
        self.n_plot_iter = n_plot_iter
        self.fig = plt.figure()
        self.camera = Camera(self.fig)
        self.prev_agent_positions = [None for i in range(self.n_agents)]

    def initialize_agents(self):
        agents = []
        for i in range(self.n_agents):
            if i < int(self.n_agents / 2):
                agent_type = "h"
            else:
                agent_type = "v"
            agent = Agent(
                position=(0, 0),
                name=str(i),
                agent_type=agent_type,
                input_dim=self.noise_dim,
                lr=self.lr,
            )  # Agent that travels horizontally in first step
            agents.append(agent)
        self.agents = agents
        self.reset_agent_positions()
        return agents

    def reset_agent_positions(self):
        for i, agent in enumerate(self.agents):
            if i < int(self.n_agents / 2):
                agent_pos = (0, (i * 2 + 1))
            else:
                agent_pos = (int(i % (self.n_agents / 2)) * 2 + 1, 0)
            agent.position = agent_pos
            agent.collided = False
            agent.done = False

    def clear_screen(self):
        os.system("clear")

    def render(self, prev_state=None):
        # self.clear_screen()

        # Plot lanes
        print("grid size: ", self.grid_dim)
        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                # For horizontal lines, i indexes vertical axis and j indexes horizontal axis
                if j % 2 == 0:
                    xmin = j
                    xmax = j + 0.5
                else:
                    xmin = j + 0.5
                    xmax = j + 1
                    
                plt.hlines(y=0.5 + i, xmin=xmin, xmax=xmax, color="b", linestyle="-")
                plt.vlines(x=0.5 + i, ymin=xmin, ymax=xmax, color="b", linestyle="-")

                # plt.vlines(x=0.5 + i, ymin=j, ymax=j + 0.5, color="b", linestyle="-")

        for i, agent in enumerate(self.agents):
            if agent.type == "h":
                color = "blue"
            else:
                color = "red"

            if agent.collided:
                plt.scatter(
                    x=agent.position[0], y=agent.position[1], c="black", marker="x"
                )
            else:
                plt.scatter(x=agent.position[0], y=agent.position[1], c=color)

            # Draw a line from prev position to current position
            if self.prev_agent_positions[i] is not None:
                plt.plot(
                    [self.prev_agent_positions[i][0], agent.position[0]],
                    [self.prev_agent_positions[i][1], agent.position[1]],
                    c=color,
                )
            self.prev_agent_positions[i] = agent.position
        plt.ylim(0, self.grid_dim - 1)
        plt.xlim(0, self.grid_dim - 1)
        plt.axis('off')

        self.camera.snap()

        """
        print("------------------------------------------------------------")
        for y_idx, row in enumerate(self.grid):
            for x_idx, col in enumerate(row):

                if self.grid[x_idx, y_idx] < 0:
                    print(self.grid[x_idx, y_idx], end="\t")
                elif self.grid[x_idx, y_idx] > 0:
                    for agent in self.agents:
                        if agent.position == (x_idx, y_idx):
                            print(agent.name, end="\t")
                else:
                    print("-", end="\t")
            print("\n")
        print("------------------------------------------------------------")
        """

    def step(self, train=True):
        self.reset_grid()
        noise = self.sample_noise()
        agent_stoch_actions, agent_actions = self.get_agent_actions(noise)
        collision_tracker = self.apply_agent_actions(agent_actions)
        agent_payoffs = self.get_payoffs(
            agent_actions, agent_stoch_actions, collision_tracker
        )
        self.current_payoffs.append(agent_payoffs)
        if train:
            self.update_agent_weights(agent_payoffs)
        else:  # display payoffs
            # print("noise: ", noise)
            avg_payoff = np.nanmean(
                np.asarray(
                    [
                        i.detach().numpy() if type(i) != float else i
                        for i in agent_payoffs
                    ]
                )
            )
            print("average payoff: ", avg_payoff)
        self.check_if_agents_are_done()

    def reset_grid(self):
        # Keep collisions
        self.grid[self.grid > 0] = 0
        # self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int8)

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
                or agent.collided
            ):
                agent.done = True

    def apply_agent_actions(self, agent_actions):
        collision_tracker = {}
        passing_grid = np.zeros((self.grid_dim, self.grid_dim), dtype=np.int8)
        passing_tracker = {}
        # Passing grid stores which points agents pass through
        # Need to detect collisions before move agents to new positions
        for i, agent in enumerate(self.agents):
            if agent.done:
                continue
            action = agent_actions[i]
            if action == 1:
                if agent.type == "v":  # vertical agent
                    passing_pos = (agent.position[0], agent.position[1] + 1)
                else:  # horizontal agent
                    passing_pos = (agent.position[0] + 1, agent.position[1])

                passing_grid[passing_pos[0], passing_pos[1]] += 1
                if passing_pos in passing_tracker:
                    passing_tracker[passing_pos].append(agent.name)
                else:
                    passing_tracker[passing_pos] = [agent.name]

        for i, agent in enumerate(self.agents):
            if agent.done:
                continue
            action = agent_actions[i]
            if action == 1:
                # Get movement direction
                if agent.type == "v":  # vertical agent
                    passing_pos = (agent.position[0], agent.position[1] + 1)

                else:  # horizontal agent
                    passing_pos = (agent.position[0] + 1, agent.position[1])

                # Collision if two players pass over same point
                if passing_grid[passing_pos[0], passing_pos[1]] > 1:
                    agent.collided = True
                    agent.position = (passing_pos[0], passing_pos[1])
                    self.grid[passing_pos[0], passing_pos[1]] -= 1
                    colliding_agents = passing_tracker[passing_pos].copy()
                    colliding_agents.remove(agent.name)
                    collision_tracker[agent.name] = colliding_agents
                # No collision
                else:
                    # Agents move 2 squares at a time
                    self.move_agent_position(agent)

        self.set_grid()
        return collision_tracker

    def move_agent_position(self, agent):
        # Two passes to catch collisions
        if agent.type == "v":  # vertical agent
            agent.position = (agent.position[0], agent.position[1] + 2)
        else:  # horizontal agent
            agent.position = (agent.position[0] + 2, agent.position[1])

    def get_payoffs(self, agent_actions, agent_stoch_actions, collision_tracker):
        # for each agent, get the current payoff
        agent_payoffs = []
        for i, agent in enumerate(self.agents):
            if agent.done:
                agent_payoffs.append(np.nan)
                continue
            stoch_action = agent_stoch_actions[i]
            if not agent.collided:  # no collision
                payoff = torch.matmul(stoch_action, self.no_collision_payoff)
            else:  # collision
                collided_idx = int(collision_tracker[agent.name][0])
                payoff = torch.matmul(
                    torch.matmul(stoch_action, self.collision_payoff),
                    agent_stoch_actions[collided_idx].detach(),
                )
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
            if agent.collided or agent.done:
                continue
            x_idx, y_idx = agent.position
            self.grid[x_idx, y_idx] += 1

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
        for i in range(self.n_agents):
            self.step(train=train)
            if display:
                self.render()

        if display:
            animation = self.camera.animate()
            if os.path.isfile("animation.mov"):
                os.remove("animation.mov")  # Opt.: os.system("rm "+strFile)
            animation.save("animation.mov")

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
        print("average_payoffs: ", self.average_payoffs.shape)
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

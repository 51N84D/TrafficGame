from comet_ml import Experiment
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import argparse
from torch.nn import init
import matplotlib.pyplot as plt
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Two-step Payoff
              go/go        go/stop        stop/go       stop/stop
          ----------------------------------------------------------
go/go     | -200, -200  |  -99, -101  |  -99, -101  |  2,   SS
go/stop   | -101, -99   |  -101, -101 |  0,   0     |  0,   SS
stop/go   | -101, -99   |     0,  0   |  -101, -101 |  0,   SS
stop/stop |   SS,  2    |    SS,  0   |  SS, 0      |  SS, SS
          ----------------------------------------------------------
"""

SS = -1


TWO_STEP_PAYOFF_1 = torch.from_numpy(
    np.asarray(
        [
            [-200, -99, -99, 2],
            [-101, -101, 0, 0],
            [-101, 0, -101, 0],
            [SS, SS, SS, SS],
        ]
    )
).float()

TWO_STEP_PAYOFF_2 = torch.from_numpy(
    np.asarray(
        [
            [-200, -101, -101, SS],
            [-99, -101, 0, SS],
            [-99, 0, -101, SS],
            [2, 0, 0, SS],
        ]
    )
).float()


PAYOFF_STEP_1_PLAYER_1 = torch.from_numpy(
    np.asarray(
        [
            [-100, 1],
            [-1, -1],
        ]
    )
).float()

PAYOFF_STEP_2_PLAYER_1 = torch.from_numpy(
    np.asarray(
        [
            [-100, 1],
            [SS, SS],
        ]
    )
).float()

PAYOFF_STEP_1_PLAYER_2 = torch.from_numpy(
    np.asarray(
        [
            [-100, -1],
            [1, -1],
        ]
    )
).float()

PAYOFF_STEP_2_PLAYER_2 = torch.from_numpy(
    np.asarray(
        [
            [-100, SS],
            [1, SS],
        ]
    )
).float()


# Define model
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden_layers,
        num_classes,
        normalize_output=True,
    ):
        super(MLP, self).__init__()
        self.model = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers - 1):
            self.model += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.model += [nn.Linear(hidden_size, num_classes)]
        self.model = nn.Sequential(*self.model)

        self.normalize = nn.Softmax(dim=-1)
        self.normalize_output = normalize_output

    def forward(self, x):
        out = self.model(x)
        if self.normalize_output:
            out = self.normalize(out)
        return out


def get_args():
    parser = argparse.ArgumentParser(description="causal-gen")
    # dataset
    parser.add_argument("--optim", type=str, nargs="+", default="adam")
    parser.add_argument("--init", type=str, default="kaiming")
    parser.add_argument("--lr1", type=float, default=0.01)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--shared_enc", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--z_size", type=int, default=16)

    return parser.parse_args()


def init_weights(net, init_type="normal", init_gain=0.02, verbose=0):
    """Initialize network weights.
    Parameters:
        net (network)     -- network to be initialized
        init_type (str)   -- the name of an initialization method:
                             normal | xavier | kaiming
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.

    """

    if not init_type:
        print("init_type is {}, defaulting to normal".format(init_type))
        init_type = "normal"
    if not init_gain:
        print("init_gain is {}, defaulting to 0.02".format(init_gain))
        init_gain = 0.02

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if verbose > 0:
        print("initialize %s with %s" % (net.__class__.__name__, init_type))
    net.apply(init_func)


def get_optimizer(net, optim_type="adam", lr=0.01):
    assert optim_type in ("adam", "sgd", "adadelta", "adagrad")
    if optim_type == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim_type == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=lr)
    elif optim_type == "adadelta":
        opt = torch.optim.Adadelta(net.parameters(), lr=lr)
    elif optim_type == "adagrad":
        opt = torch.optim.Adagrad(net.parameters(), lr=lr)
    return opt

"""

def sample_z(mean=0, std=1, size=16, batch_size=1):
    return torch.from_numpy(
        np.random.normal(mean, std, size=(batch_size, size))
    ).float()
"""


def sample_z(mean=0, std=1, size=16, batch_size=1):
    return torch.from_numpy(
        np.tile(np.random.binomial(n=1, p=0.5, size=(1)), (batch_size, size))
    ).float()


def make_plot(
    payoffs_1,
    payoffs_2,
    optim=None,
    args=None,
    fig=None,
    ax=None,
    plot_dir="./plots",
    alpha=1.0,
):
    plot_dir = Path(plot_dir).resolve()
    plot_dir.mkdir(exist_ok=True, parents=True)

    fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    for i in range(payoffs_1.shape[0]):
        ax1[0].plot(payoffs_1[i, :])

    for i in range(payoffs_2.shape[0]):
        ax1[1].plot(payoffs_2[i, :])

    if args is not None:
        ax1[0].title.set_text(f"Payoff 1, lr {args.lr1}")
        ax1[1].title.set_text(f"Payoff 2, lr {args.lr2}")

    for i in ax1:
        i.set_ylim([-25, 5])

    fig1.savefig(plot_dir / f"{optim}_{args.n_runs}_runs.png")

    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    markers, caps, bars = ax.errorbar(
        x=np.arange(payoffs_1.shape[1]),
        y=np.mean(payoffs_1 + payoffs_2, axis=0),
        yerr=np.std(payoffs_1 + payoffs_2, axis=0),
        label=optim,
    )

    [bar.set_alpha(alpha * 0.1) for bar in bars]

    ax.legend()
    ax.set_ylim([-40, 5])

    fig.savefig(plot_dir / "plot.png")

    return fig, ax


def train():
    args = get_args()

    # comet logging
    if args.comet:
        exp = Experiment(project_name="games", auto_metric_logging=False)
        exp.add_tag(args.optim)
        exp.add_tag(args.init)
        exp.add_tag(f"lr1 {args.lr1}")
        exp.add_tag(f"lr2 {args.lr2}")

    else:
        exp = None

    fig, ax = None, None

    for i_optim, optim in enumerate(args.optim):
        payoff_1 = []
        payoff_2 = []
        with tqdm(total=args.n_runs * args.n_iter) as pbar:
            for i in range(args.n_runs):
                if args.shared_enc:
                    latent_player = MLP(
                        input_size=args.z_size * 2,
                        hidden_size=8,
                        n_hidden_layers=4,
                        num_classes=2,
                        normalize_output=True,
                    )
                else:
                    latent_player = MLP(
                        input_size=args.z_size,
                        hidden_size=8,
                        n_hidden_layers=4,
                        num_classes=4,
                        normalize_output=True,
                    )
                init_weights(net=latent_player, init_type=args.init)
                opt_latent = get_optimizer(latent_player, optim, args.lr1)

                if not args.shared_enc:
                    player_1 = MLP(
                        input_size=4, hidden_size=4, n_hidden_layers=4, num_classes=2
                    )
                    player_2 = MLP(
                        input_size=4, hidden_size=4, n_hidden_layers=4, num_classes=2
                    )
                    init_weights(net=player_1, init_type=args.init)
                    init_weights(net=player_2, init_type=args.init)
                    opt_1 = get_optimizer(player_1, optim, args.lr1)
                    opt_2 = get_optimizer(player_2, optim, args.lr2)

                payoff_1_local = []
                payoff_2_local = []

                for j in range(args.n_iter):
                    z1 = sample_z(size=args.z_size, batch_size=args.batch_size)
                    z2 = sample_z(size=args.z_size, batch_size=args.batch_size)

                    if args.shared_enc:
                        z1_p = torch.cat((z1, torch.zeros((args.batch_size, args.z_size))), dim=-1)
                        z2_p = torch.cat((z2, torch.zeros((args.batch_size, args.z_size))), dim=-1)

                        z1_q = torch.cat((z1, torch.ones((args.batch_size, args.z_size))), dim=-1)
                        z2_q = torch.cat((z2, torch.ones((args.batch_size, args.z_size))), dim=-1)

                        p1 = latent_player(z1_p)
                        q1 = latent_player(z1_q)

                        p2 = latent_player(z2_p)
                        q2 = latent_player(z2_q)

                    else:
                        z1 = latent_player(z1)
                        z2 = latent_player(z2)

                        p1 = player_1(z1)
                        q1 = player_2(z1)

                        p2 = player_1(z2)
                        q2 = player_2(z2)   

                    loss_1 = -(
                        (
                            torch.bmm(
                                p1.unsqueeze(1),
                                torch.matmul(
                                    PAYOFF_STEP_1_PLAYER_1.detach(), q1.T.detach()
                                ).T.unsqueeze(-1),
                            )
                        )
                        + torch.bmm(
                            p2.unsqueeze(1),
                            torch.matmul(
                                PAYOFF_STEP_2_PLAYER_1.detach(), q2.T.detach()
                            ).T.unsqueeze(-1),
                        )
                    ).mean()

                    loss_2 = -(
                        (
                            torch.bmm(
                                p1.unsqueeze(1).detach(),
                                torch.matmul(
                                    PAYOFF_STEP_1_PLAYER_2.detach(), q1.T
                                ).T.unsqueeze(-1),
                            )
                        )
                        + torch.bmm(
                            p2.unsqueeze(1).detach(),
                            torch.matmul(
                                PAYOFF_STEP_2_PLAYER_2.detach(), q2.T
                            ).T.unsqueeze(-1),
                        )
                    ).mean()

                    latent_player.zero_grad()

                    if not args.shared_enc:
                        player_1.zero_grad()
                        player_2.zero_grad()

                    loss = loss_1 + loss_2
                    loss.backward()
                    opt_latent.step()

                    if not args.shared_enc:
                        opt_1.step()
                        opt_2.step()

                    if exp is not None:
                        exp.log_metric("payoff 1", -loss_1.item(), step=j)
                        exp.log_metric("payoff 2", -loss_2.item(), step=j)

                    payoff_1_local.append(-loss_1.item())
                    payoff_2_local.append(-loss_2.item())
                    pbar.update(1)

                payoff_1.append(payoff_1_local)
                payoff_2.append(payoff_2_local)

        payoff_1 = np.asarray(payoff_1)
        payoff_2 = np.asarray(payoff_2)

        fig, ax = make_plot(
            payoff_1,
            payoff_2,
            optim=optim,
            args=args,
            fig=fig,
            ax=ax,
            alpha=1.0 / (i_optim + 1),
        )
        print("------------------------")
        print(np.rint(payoff_1[:, -1]))
        print(np.rint(payoff_2[:, -1]))
        print("------------------------")


if __name__ == "__main__":
    train()
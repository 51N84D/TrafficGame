from torch.nn import init
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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

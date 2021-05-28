import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from utils import init_weights, get_optimizer, make_plot
from models import MLP

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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

SS = -10

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


def get_args():
    parser = argparse.ArgumentParser(description="two-step-traffic-game")
    # dataset
    parser.add_argument(
        "--optim",
        type=str,
        nargs="+",
        default="adam",
        help="list of optims to iterate over",
    )
    parser.add_argument("--init", type=str, default="kaiming")
    parser.add_argument("--lr1", type=float, default=0.01)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--comet", action="store_true", help="Log experiment to comet")
    parser.add_argument(
        "--shared_enc",
        action="store_true",
        help="Use single network for both players",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--z_size", type=int, default=16, help="size of latent vector")
    parser.add_argument(
        "--process_latent",
        action="store_true",
        help="process latent vector w/ NN before feeding to players",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="normal",
        help="noise distribution {'normal' or 'bernoulli'}",
    )
    parser.add_argument("--n_eval", type=int, default=10, help='number of samples to evaluate on')


    return parser.parse_args()


def sample_z(args, size=16, batch_size=1):
    if args.noise == "normal":
        return sample_z_normal(size=size, batch_size=batch_size)
    else:  # args.noise == 'bernoulli':
        return sample_z_bernoulli(size=size, batch_size=batch_size)


def sample_z_normal(mean=0, std=1, size=16, batch_size=1):
    return torch.from_numpy(
        np.random.normal(mean, std, size=(batch_size, size))
    ).float()


def sample_z_bernoulli(size=16, batch_size=1):
    return (
        torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(batch_size)))
        .float()
        .unsqueeze(-1)
        .repeat(1, size)
    )


def get_actions(args, player_1, player_2, latent_player):
    z1 = sample_z(args, size=args.z_size, batch_size=args.batch_size)
    z2 = sample_z(args, size=args.z_size, batch_size=args.batch_size)

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
        if args.process_latent:
            z1 = latent_player(z1)
            z2 = latent_player(z2)

        p1 = player_1(z1)
        q1 = player_2(z1)

        p2 = player_1(z2)
        q2 = player_2(z2)

    return p1, p2, q1, q2


def get_loss(p1, p2, q1, q2):
    # Second loss should depend on first action. We can sample with some probability

    loss_p1 = -torch.bmm(
        p1.unsqueeze(1),
        torch.matmul(PAYOFF_STEP_1_PLAYER_1, q1.T.detach()).T.unsqueeze(-1),
    )

    # Sample 2nd loss matrix from bernoulli
    # x = np.random.binomial(n=p1.shape[0], p=p1[:, 0].detach())

    p2_aug_payoff = torch.matmul(
        p1,
        torch.cat(
            (PAYOFF_STEP_1_PLAYER_1.unsqueeze(0), PAYOFF_STEP_2_PLAYER_1.unsqueeze(0)),
            dim=0,
        ).reshape(2, 4),
    ).reshape(-1, 2, 2)

    loss_p2 = -torch.bmm(p2.unsqueeze(1), torch.bmm(p2_aug_payoff, q2.unsqueeze(-1)))

    """
    loss_p2 = -torch.bmm(
        p2.unsqueeze(1),
        torch.matmul(PAYOFF_STEP_2_PLAYER_1.detach(), q2.T.detach()).T.unsqueeze(-1),
    ).mean()
    """

    loss_1 = (loss_p1 + loss_p2).mean()

    loss_q1 = -torch.bmm(
        p1.unsqueeze(1).detach(),
        torch.matmul(PAYOFF_STEP_1_PLAYER_2.detach(), q1.T).T.unsqueeze(-1),
    ).mean()

    q2_aug_payoff = torch.matmul(
        q1,
        torch.cat(
            (PAYOFF_STEP_1_PLAYER_2.unsqueeze(0), PAYOFF_STEP_2_PLAYER_2.unsqueeze(0)),
            dim=0,
        ).reshape(2, 4),
    ).reshape(-1, 2, 2)

    loss_q2 = -torch.bmm(p2.unsqueeze(1), torch.bmm(q2_aug_payoff, q2.unsqueeze(-1)))

    """
    loss_q2 = -torch.bmm(
        p2.unsqueeze(1).detach(),
        torch.matmul(PAYOFF_STEP_2_PLAYER_2.detach(), q2.T).T.unsqueeze(-1),
    ).mean()
    """

    loss_2 = (loss_q1 + loss_q2).mean()

    return loss_1, loss_2, loss_1 + loss_2


def create_models(args):
    player_1 = None
    player_2 = None
    latent_player = None

    if args.shared_enc:
        latent_player = MLP(
            input_size=args.z_size * 2,
            hidden_size=8,
            n_hidden_layers=4,
            num_classes=2,
            normalize_output=True,
        )
    elif args.process_latent:
        latent_player = MLP(
            input_size=args.z_size,
            hidden_size=8,
            n_hidden_layers=4,
            num_classes=args.z_size,
            normalize_output=True,
        )

    if not args.shared_enc:
        player_1 = MLP(
            input_size=args.z_size, hidden_size=4, n_hidden_layers=4, num_classes=2
        )
        player_2 = MLP(
            input_size=args.z_size, hidden_size=4, n_hidden_layers=4, num_classes=2
        )

    return player_1, player_2, latent_player


def train():
    args = get_args()

    fig, ax = None, None

    for i_optim, optim in enumerate(args.optim):
        payoff_1 = []
        payoff_2 = []

        with tqdm(total=args.n_runs * args.n_iter) as pbar:
            for i in range(args.n_runs):
                player_1, player_2, latent_player = create_models(args)

                if args.process_latent or args.shared_enc:
                    init_weights(net=latent_player, init_type=args.init)
                    opt_latent = get_optimizer(latent_player, optim, args.lr1)

                if not args.shared_enc:
                    init_weights(net=player_1, init_type=args.init)
                    init_weights(net=player_2, init_type=args.init)
                    opt_1 = get_optimizer(player_1, optim, args.lr1)
                    opt_2 = get_optimizer(player_2, optim, args.lr2)

                payoff_1_local = []
                payoff_2_local = []

                for j in range(args.n_iter):

                    p1, p2, q1, q2 = get_actions(
                        args, player_1, player_2, latent_player
                    )

                    # Manually set for testing
                    """
                    p1 = torch.from_numpy(np.asarray([0.5, 0.5])).unsqueeze(0).repeat(args.batch_size, 1).float()
                    p2 = torch.from_numpy(np.asarray([0.5, 0.5])).unsqueeze(0).repeat(args.batch_size, 1).float()

                    q1 = torch.from_numpy(np.asarray([0.5, 0.5])).unsqueeze(0).repeat(args.batch_size, 1).float()
                    q2 = torch.from_numpy(np.asarray([0.5, 0.5])).unsqueeze(0).repeat(args.batch_size, 1).float()
                    """

                    loss_1, loss_2, loss = get_loss(p1, p2, q1, q2)

                    if args.process_latent or args.shared_enc:
                        latent_player.zero_grad()

                    if not args.shared_enc:
                        player_1.zero_grad()
                        player_2.zero_grad()

                    loss.backward()

                    if args.process_latent or args.shared_enc:
                        opt_latent.step()

                    if not args.shared_enc:
                        opt_2.step()
                        opt_1.step()

                    payoff_1_local.append(-loss_1.item())
                    payoff_2_local.append(-loss_2.item())
                    pbar.update(1)

                    # print('loss: ', loss)

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

        count_coord = 0
        for run_idx in range(payoff_1.shape[0]):
            if (payoff_1[run_idx, -1] + payoff_2[run_idx, -1]).mean() > SS:
                count_coord += 1

        # Evaluation
        print(f"Fraction of coordination: {count_coord}/{payoff_1.shape[0]}")
        p_count = 0  # number of times player 1 goes
        q_count = 0  # number of times player 2 goes
        for j in range(args.n_eval):
            p, _, q, _ = get_actions(args, player_1, player_2, latent_player)
            print(
                f"p: {np.argmax(p[0].detach().numpy())}, q: {np.argmax(q[0].detach().numpy())}"
            )
            p_count += np.argmax(p[0].detach().numpy())
            q_count += np.argmax(q[0].detach().numpy())

        print(f"TOTALS: p={p_count}/{args.n_eval}, q={q_count}/{args.n_eval} ")
        # print(np.rint(payoff_1[:, -1]))
        # print(np.rint(payoff_2[:, -1]))
        # print("------------------------")


if __name__ == "__main__":
    train()
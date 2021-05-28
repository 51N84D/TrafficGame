import argparse
from gridgame import GridGame


def get_args():
    parser = argparse.ArgumentParser(description="grid-traffic-game")
    parser.add_argument(
        "--n-agents",
        type=int,
        default=1,
        help="Number of agents on each side (i.e. total agents = n_agents * 2)",
    )
    parser.add_argument("--noise-dim", type=int, default=16)
    parser.add_argument("--n-train-iter", type=int, default=1000)
    parser.add_argument("--n-save-iter", type=int, default=100)
    parser.add_argument("--n-plot-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--load-dir", type=str)
    parser.add_argument("--evaluate", action="store_true", help="evaluate model")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    env = GridGame(
        n_agents_one_side=args.n_agents,
        noise_dim=args.noise_dim,
        save_dir=args.save_dir,
        load_dir=args.load_dir,
        n_train_iter=args.n_train_iter,
        n_save_iter=args.n_save_iter,
        n_plot_iter=args.n_plot_iter,
        resume=args.evaluate,
        lr=args.lr,
    )

    if args.evaluate:
       env.test()
    else:
       env.train()

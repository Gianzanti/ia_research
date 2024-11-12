import argparse

from check_env import show_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        help="Algorithm to use in training",
        choices=["PPO", "SAC", "DDPG", "TD3", "A2C"],
        default="PPO",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Mode to run the program",
        choices=["train", "test", "check"],
        default="train",
    )
    parser.add_argument(
        "-d",
        "--model",
        type=str,
        help="Path to the model file to test (just the number)",
        default="100000",
    )

    args = parser.parse_args()
    print(args)

    if args.mode == "check":
        print("Checking environment...")
        show_env()

    # main()

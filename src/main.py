import argparse

from robotis_env.robotis_env import RobotisEnv  # noqa: F401

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
        choices=["train", "test", "check", "check_model", "keep", "transfer"],
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
    # print(args)

    if args.mode == "check":
        print("Checking environment...")
        from checking import show_env

        show_env()
    elif args.mode == "check_model":
        print("Checking model...")
        from checking import check_model

        check_model()
    elif args.mode == "train":
        print("Training...")
        from training import train

        train(args.algo)
    elif args.mode == "test":
        print("Testing...")
        from training import test

        test(args.algo, args.model)
    
    elif args.mode == "keep":
        print("Training...")
        from training import keep_training

        keep_training(args.algo, args.model)
        

    elif args.mode == "transfer":
        print("Transfering...")
        from training import transfer

        transfer(args.algo, args.model)
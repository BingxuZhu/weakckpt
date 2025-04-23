import argparse
from models.model_deepspeed import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training with WeakCkpt + DeepSpeed")
    parser.add_argument(
        '--deepspeed',
        action='store_true',
        help='Flag to indicate running under deepspeed launcher'
    )
    args = parser.parse_args()

    train()
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_val_split', type=float, default=0.95)
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--rnn_type', type=str, default='lstm')
parser.add_argument('--hidden_units', type=int, default=300)
parser.add_argument('--clip_norm', type=int, default=5.0)

args = parser.parse_args()

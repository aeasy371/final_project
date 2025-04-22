#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
To run mpc_autograd_cnn example:

$ python examples/mpc_autograd_cnn/launcher.py

To run mpc_linear_svm example on AWS EC2 instances:

$ python3 scripts/aws_launcher.py \
	--ssh_key_file=$HOME/.aws/fair-$USER.pem \
	--instances=i-038dd14b9383b9d79,i-08f057b9c03d4a916 \
	--aux_files=examples/mpc_autograd_cnn/mpc_autograd_cnn.py \
	examples/mpc_autograd_cnn/launcher.py
"""

import argparse
import logging
import os

from multiprocess_launcher import (
	MultiProcessLauncher,
)
import crypten


parser = argparse.ArgumentParser(description="CrypTen Autograd CNN Training")


def validate_world_size(world_size):
	world_size = int(world_size)
	if world_size < 2:
		raise argparse.ArgumentTypeError(f"world_size {world_size} must be > 1")
	return world_size


parser.add_argument(
	"--world_size",
	type=validate_world_size,
	default=2,
	help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
	"--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
	"--lr",
	"--learning-rate",
	default=0.001,
	type=float,
	metavar="LR",
	help="initial learning rate",
)
parser.add_argument(
	"-b",
	"--batch-size",
	default=32,
	type=int,
	metavar="N",
	help="mini-batch size (default: 5)",
)
parser.add_argument(
	"--print-freq",
	"-p",
	default=50,
	type=int,
	metavar="PF",
	help="print frequency (default: 5)",
)
parser.add_argument(
	"--num-samples",
	"-n",
	default=100,
	type=int,
	metavar="N",
	help="num of samples used for training (default: 100)",
)


def _run_CIFAR_experiment(args):
	level = logging.INFO
	if "RANK" in os.environ and os.environ["RANK"] != "0":
		level = logging.CRITICAL
	logging.getLogger().setLevel(level)
	logging.basicConfig(
		level=level,
		format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
	)
	# pyre-fixme[21]: Could not find module `mpc_autograd_cnn`.
	from MPC_Crypten_autograd import run_mpc_autograd_fine_tuning  # @manual

	import torch
	from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


	# Load data
	cifar10 = torch.load('cifar10_swin_features.pt')
	X = cifar10['features']
	y = cifar10['labels'].long()

	# Dataset
	full_dataset = TensorDataset(X, y)
	batch_size=128

	# Split sizes
	total = len(full_dataset)
	train_size = int(0.8 * total)
	val_size = int(0.1 * total)
	test_size = total - train_size - val_size

	train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=batch_size)
	test_loader = DataLoader(test_set, batch_size=batch_size)

	run_mpc_autograd_fine_tuning(
		train_loader,
		val_loader,
		test_loader,
		num_epochs=args.epochs,
		learning_rate=args.lr,
		batch_size=args.batch_size,
		print_freq=args.print_freq,
		num_samples=args.num_samples,
	)

def _run_MNIST_experiment(args):
	level = logging.INFO
	if "RANK" in os.environ and os.environ["RANK"] != "0":
		level = logging.CRITICAL
	logging.getLogger().setLevel(level)
	logging.basicConfig(
		level=level,
		format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
	)
	# pyre-fixme[21]: Could not find module `mpc_autograd_cnn`.
	from MPC_Crypten_autograd import run_mpc_autograd_fine_tuning  # @manual

	import torch
	from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
	from torchvision import datasets, transforms
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	# Load MNIST train/test datasets
	train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
	test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

	# DataLoaders for batching
	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

	run_mpc_autograd_fine_tuning(
		train_loader,
		test_loader,
		test_loader,
		num_epochs=args.epochs,
		learning_rate=args.lr,
		batch_size=args.batch_size,
		print_freq=args.print_freq,
		num_samples=args.num_samples,
	)


def main(run_experiment):
	args = parser.parse_args()
	# run multiprocess by default
	launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
	launcher.start()
	launcher.join()
	launcher.terminate()


if __name__ == "__main__":
	crypten.init()
	main(_run_CIFAR_experiment)
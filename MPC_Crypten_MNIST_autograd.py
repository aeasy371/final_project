""" This code is based off the example from Crypten's github page"""
""" 
	original basis code is found at:
	https://github.com/facebookresearch/CrypTen/blob/main/examples/mpc_autograd_cnn/mpc_autograd_cnn.py
"""


import crypten
import crypten.nn as nn
import crypten.communicator as comm
import crypten.communicator
import torch
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


import io	# needed to save the crypten models


def run_mpc_autograd_fine_tuning(train_dataloader, val_dataloader, test_dataloader, context_manager=None, num_epochs=3, learning_rate=0.001, batch_size=32, print_freq=5, num_samples=100):

	rank = comm.get().get_rank()

	# For MNIST, this model is more complex because we are not using a feature extractor on it.
	crypten_model = crypten.nn.Sequential(
		# crypten.nn.Flatten(),                     # Flatten 28x28 -> 784
		crypten.nn.Linear(784, 256),
		crypten.nn.ReLU(),
		crypten.nn.Linear(256, 128),
		crypten.nn.ReLU(),
		crypten.nn.Linear(128, 10)               # 10 output classes for MNIST
	)

	# Encrypt the model for MPC
	crypten_model.train()
	crypten_model.encrypt()
	
	
	print('Training now')
	train_encrypted_model(
		train_loader=train_dataloader, encrypted_model=crypten_model, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, print_freq=print_freq,
		val_dataloader=val_dataloader, evaluation=test_encrypted_model, evaluation_intervals=1
	)
	for i in range(5):
		test_encrypted_model(
			test_dataloader, crypten_model
		)

def train_encrypted_model(train_loader, encrypted_model, num_epochs, learning_rate, batch_size, print_freq, val_dataloader, evaluation=None, evaluation_intervals=10):
	rank = comm.get().get_rank()
	loss = crypten.nn.loss.CrossEntropyLoss()
	encrypted_model = encrypted_model
	# optimizer = crypten.optim.SGD(encrypted_model.parameters(), learning_rate, weight_decay=0.02)

	num_samples = len(train_loader)
	# num_samples = x_encrypted.size(0)
	# label_eye = torch.eye(2)


	for epoch in range(num_epochs):
		# losses = torch.zeros(int(num_samples / batch_size))
		encrypted_model.train()
		learning_rate = 0.01
		optimizer = crypten.optim.SGD(encrypted_model.parameters(), learning_rate)
		total_correct = 0
		total = 0
		# only print from rank 0 to avoid duplicates for readability
		# if rank == 0:
			# print(f'Epoch {epoch} in progress:')

		# for j in range(0, num_samples, batch_size):
		j = 0
		running_loss = 0.0
		for video, label in train_loader:
			video = video.view(-1, 28*28)
			encrypted_model.zero_grad()
			# encrypt tensors
			video = crypten.cryptensor(video, requires_grad=True)

			label = one_hot(label, num_classes=10)
			label = crypten.cryptensor(label, requires_grad=True)
			
			video.requires_grad = True
			label.requires_grad = True		# maybe turn this off?
			# print("vpted[start:end]

			# perform forward pass
			output = encrypted_model(video)
			# printabel.shape)
			loss_value = loss(output, label)

			# backprop
			loss_value.backward()
			optimizer.step()

			# log progress {loss_value.get_plain_text().item():.4f}')
			running_loss += loss_value.get_plain_text().item()
			# compute accuracy every epoch

			pred = output.get_plain_text().argmax(-1)
			y_true = label.get_plain_text().argmax(-1)
			correct = pred == y_true
			correct_count = correct.sum(0, keepdim=True).float()
			total_correct += correct_count.item()
			total += output.get_plain_text().size(0)
		accuracy = total_correct / total
		if rank == 0:
			print(
				f'Epoch {epoch} completed: '
				f'Loss {running_loss / len(train_loader):.4f} Accuracy {accuracy:.2f}'
			)
			with open(f'training_outputs/training_{learning_rate}.txt', 'w+') as f:
				f.write(f'Epoch {epoch} completed: \n' + 
				f'Loss {running_loss / len(train_loader):.4f} Accuracy {accuracy:.2f}\n')

		if epoch % evaluation_intervals == 0 or epoch == num_epochs - 1:
			if evaluation:
				evaluation(val_dataloader, encrypted_model)

		crypten.save(encrypted_model, f'encrypted_models/MNIST_{learning_rate}_{epoch+1}.pth')
		encrypted_model.encrypt()


def test_encrypted_model(test_dataloader, model):
	model.eval()
	loss = crypten.nn.loss.CrossEntropyLoss()
	total_correct = 0
	total = 0
	running_loss = 0.0
	rank = comm.get().get_rank()
	with crypten.no_grad():
		for videos, labels in test_dataloader:
			videos = videos.view(-1, 28*28)
			videos = crypten.cryptensor(videos)
			labels = one_hot(labels, num_classes=10)
			labels = crypten.cryptensor(labels)

			outputs = model(videos)
			loss_value = loss(outputs, labels)

			pred = outputs.get_plain_text().argmax(-1)
			y_true = labels.get_plain_text().argmax(-1)
			correct = pred == y_true
			correct_count = correct.sum(0, keepdim=True).float()
			total_correct += correct_count.item()
			total += outputs.get_plain_text().size(0)

			running_loss += loss_value.get_plain_text().item()
		model.zero_grad()
		
	if comm.get().get_rank() == 0:
		print(f'Test loss: {running_loss:.4f}')
		print(f'Test Accuracy: {(total_correct / total)*100:.2f}%')

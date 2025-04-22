import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoFeatureExtractor, SwinModel
from PIL import Image
# from timm import create_model
from tqdm import tqdm

def create_featureset(dataset_name: str):
	# Parameters
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	batch_size = 64
	save_path = f"{dataset_name}_swin_features.pt"

	# Load HuggingFace Swin
	model_name = "microsoft/swin-tiny-patch4-window7-224"
	extractor = AutoFeatureExtractor.from_pretrained(model_name)
	model = SwinModel.from_pretrained(model_name)
	model.eval().to(device)

	# Typical ViT preprocessing (same as Hugging Face ViT config)
	transform = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])


	# Load CIFAR dataset
	if dataset_name == 'cifar10':
		dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	elif dataset_name == 'cifar100':
		dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	# Store features + labels
	all_features = []
	all_labels = []

	with torch.no_grad():
		for imgs, labels in tqdm(dataloader):
			imgs = imgs.to(device)
			features = model(imgs)
			features = features.pooler_output
			all_features.append(features.cpu())
			all_labels.append(labels)

	# Save to file
	features_tensor = torch.cat(all_features)
	labels_tensor = torch.cat(all_labels)
	torch.save({'features': features_tensor, 'labels': labels_tensor}, save_path)

	print(f"Saved features to {save_path}")


create_featureset('cifar10')
create_featureset('cifar100')

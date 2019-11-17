import os
import sys
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from mydatasets import load_dataset
from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from mymodels import MyCNN


torch.manual_seed(0)
if torch.cuda.is_available():
	torch.cuda.manual_seed(0)

df=pd.read_csv("../CheXpert-v1.0-small/filtered_train.csv", sep=',')
df['Patient'] = df.apply (lambda row: row['Path'].split('/')[2].replace('patient',''), axis=1)
cols = df.columns.tolist()
cols = cols[-1:]+cols[:-1]
df =df[cols]

# Path for saving model
PATH_OUTPUT = "best_CNNmodel/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

# Some parameters
MODEL_TYPE = 'CNN'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task
model = MyCNN()
save_file = 'modelCNN.pth'
NUM_EPOCHS = 10
BATCH_SIZE = 50
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

train_dataset, valid_dataset, test_dataset = load_dataset(df,"Frontal", MODEL_TYPE)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, save_file))

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

out_df = pd.DataFrame(test_results,columns=['Expected','Output'])
out_df.to_csv("results_cnn.csv")

class_names = ['No Pheurial', 'Pheurial']
plot_confusion_matrix(test_results, class_names)

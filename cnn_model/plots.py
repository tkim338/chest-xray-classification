import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	x_epochs = range(len(train_losses))

	fig1 = plt.figure(1)
	plt.plot(x_epochs, train_losses, label="TRAIN")
	plt.plot(x_epochs, valid_losses, label = "VALID")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss Curve")
	plt.legend()
	plt.grid(True)
	fig1.savefig('Losses/losses_cnn25000.png')

	fig2 = plt.figure(2)
	plt.plot(x_epochs, train_accuracies, label="TRAIN")
	plt.plot(x_epochs, valid_accuracies, label = "VALID")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy Curve")
	plt.legend(loc='lower right')
	plt.grid(True)
	fig2.savefig('Accuracies/accuracies_cnn25000.png')

	pass


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	y_val = np.array(results)
	# print(y_val)
	y_true = y_val[:,0]
	y_pred = y_val[:,1]

	conf_mat = confusion_matrix(y_true,y_pred)
	conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

	fig, ax = plt.subplots()
	im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(conf_mat.shape[1]),yticks=np.arange(conf_mat.shape[0]),
			xticklabels=class_names, yticklabels=class_names,
			title="Normalized Confusion Matrix",ylabel='True',xlabel='Predicted')

	plt.setp(ax.get_xticklabels(), rotation=60, ha="right",rotation_mode="anchor")
	# Loop over data dimensions and create text annotations.
	#TWO DECIMALS
	fmt = '.2f'
	#Differentiate between black and white backgrounds
	thresh = conf_mat.max() / 2.
	for i in range(conf_mat.shape[0]):
		for j in range(conf_mat.shape[1]):
			ax.text(j, i, format(conf_mat[i, j], fmt),
	        		ha="center", va="center",
					#Differentiate between black and white backgrounds
	        		color="white" if conf_mat[i, j] > thresh else "black")
	fig.tight_layout()
	fig.savefig('ConfusionMatrix/cm_cnn25000.png')

	pass

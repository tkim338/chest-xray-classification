import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, jaccard_score, accuracy_score, f1_score

def calc_scores(in_path):

	# Open files
	dataframe = pd.read_csv(in_path)
	#out = open(out_path, "w+")

	# get labels
	y_true = dataframe["Expected"]
	y_pred = dataframe["Output"]

	# compute averages
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	jaccard = jaccard_score(y_true, y_pred)
	accuracy = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)

	print("Scores for", in_path)
	print("\tAccuracy:", accuracy)
	print("\tPrecision:", precision)
	print("\tRecall:", recall)
	print("\tJaccard:", jaccard)
	print("\tF1 Score:", f1, "\n")

	#out.write()

	#out.close()

calc_scores("../knn_model/results_knn.csv")
calc_scores("../labelspread/label_spreading_results.csv")
calc_scores("../cnn_model/results_cnn.csv")
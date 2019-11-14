import numpy as np
import random
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score

# taken from here: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


gmm_path = "../gmm/gmm_predictions_20_clusters.csv"
kmeans_path = "../kmeans/kmeans_predictions_20_clusters.csv"
out_path = "./evaluations.txt"

# Open files
gmm_file = open(gmm_path)
kmeans_file = open(kmeans_path)
out = open(out_path, "w+")

# skip headers
gmm_file.readline()
kmeans_file.readline()

# take 100 random samples of 1000 data cases and average the results
total_gmm_purity = 0
total_kmeans_purity = 0
total_gmm_mutual_info = 0
total_kmeans_mutual_info = 0
for i in range(100):
	indices = np.array(random.sample(range(0, 108569), 1000))
	truth = np.zeros(1000)
	pred_gmm = np.zeros(1000)
	pred_kmeans = np.zeros(1000)

	# populate truth and pred arrays
	row_count = 0
	counter = 0
	for gmm_row in gmm_file:

		# get rows from other files
		kmeans_row = kmeans_file.readline()

		if row_count in indices:

			# Set truth array (0 = no finding, 1 = pleural effusion)
			if gmm_row.split(',')[5] == "1":
				truth[counter] = 0
			else:
				truth[counter] = 1

			# Set predictions
			pred_gmm[counter] = gmm_row.split(',')[7]
			pred_kmeans[counter] = kmeans_row.split(',')[7]

			# increment count
			counter += 1

		row_count += 1

	# get purities
	total_kmeans_purity += purity_score(truth, pred_gmm)
	total_gmm_purity += purity_score(truth, pred_kmeans)

	# get mutual info
	total_kmeans_mutual_info += normalized_mutual_info_score(truth, pred_kmeans)
	total_gmm_mutual_info += normalized_mutual_info_score(truth, pred_gmm)

	print("Completed iteration " + str(i + 1) + " of 100")

# compute averages
gmm_purity = total_gmm_purity / 100.0
kmeans_purity = total_kmeans_purity / 100.0
gmm_mutual_info = total_gmm_mutual_info / 100.0
kmeans_mutual_info = total_kmeans_mutual_info / 100.0

out.write("Purity:\n")
out.write("    GMM => " + str(gmm_purity) + "\n")
out.write("    KMeans => " + str(kmeans_purity) + "\n")
out.write("Normalized Mutual Information:\n")
out.write("    GMM => " + str(gmm_mutual_info) + "\n")
out.write("    KMeans => " + str(kmeans_mutual_info) + "\n")

out.close()
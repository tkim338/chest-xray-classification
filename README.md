# chest-xray-classification

To run the CNN model, go inside cnn_model and run python train_CNN.py
If you want to change some of the parameters, change the following as follow:

I recommend for these two to be 10 Epochs, and 32 Batch size, but feel free to change them.
train.py: NUM_EPOCHS (line 33),
			BATCH_SIZE (line 34)

In mymodels.py you could change the structure of the CNN.

In mydatasets.py, the images are being gathered as np arrays. 
Change the amount of training + valid: max_img (line 30), testing: size_test (line 60)

IMPORTANT!
plots.py: Losses (line 21), Accuracies (line 31), Confusion Matrix (line 68)
			
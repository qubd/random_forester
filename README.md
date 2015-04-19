random_forester
===============

Minimal random forest classifier in Python.

Run forester.py to do classification of quantitative data. Training data goes in training.csv, testing data goes in validation.csv, and new data to be classified goes in prediction.csv. The training and testing data given here are both Fischer's complete iris flower data set, and the prediction data is a sample of nine individuals, three of each species, from that data set, with the entry in the species column removed.

Each of the three data files must consist of comma separated values. Variable names go in the first row, and class names in the last column of the training and testing data. The three data sets must share the same arbitrary number of variables, and all must be quantitative save for the class names, which should not be present in prediction.csv. Adjusting the parameters of the forest (number of decision trees, decision tree depth, and sample size) can be done by changing the values in the build_forest call.

Future work: Categorical variables, automated cross-validation.

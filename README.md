random_forester
===============

Random forest classifier in Python.

Use forester.py to do classification of quantitative data. Everything is written from scratch, so no scientific packages are required. The training data given here is [Fisher's iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set), and the prediction data is a sample of nine individuals from that same data set, three of each species. The correct classifications are setosa for the first three, versicolor for next three, and virginica for the last three.

To build a random forest, begin by loading training data from a CSV file. This CSV must consist of comma separated values with no extra spacing. The first row should contain variable names, and the last column should contain the classes. The class variable must be the only variable in the data set which is not quantitative. Some information about the data set will be output as a sanity check.

```
>>>iris_training_set = DataCSV.from_training_data('iris_flowers.csv')
Data for 150 individuals with 4 quantitative variables.
Classes: ['I.versicolor', 'I.virginica', 'I.setosa']
```

Now that the training data has been parsed, it can be used to train a random forest. Three additional arguments need to be provided: the number of trees in the random forest, the maximum depth of the trees, and the size of the random samples used to train the decision trees. The line below creates a random forest with 100 decision trees of depth at most 5, using random samples (taken with replacement) of size 40.

```
>>>iris_forest = RandomForest(iris_training_set, 100, 5, 40)
```

To validate the random forest model, call the `cross_validate` routine, which takes a positive integer k > 1 as an argument, and performs k-fold cross validation, using the same sample size and maximum tree depth used to construct the model. The number of correctly classified individuals in each fold is reported, as well as the overall rate of correct classification. Note every time the model is cross validated, the training data is randomly reordered before being partitioned into folds. This way repeated calls can yield new information.

```
>>>iris_forest.cross_validate(3)
Fold 1: 48 of 50 individuals classified correctly.
Fold 2: 49 of 50 individuals classified correctly.
Fold 3: 46 of 50 individuals classified correctly.
Overall success rate: 95.3%
```

In order to classify new data, put it into a CSV file with the same format as the training data. Of course, the final column corresponding to the classes will not be present. To use the random forest to classify the individuals and append the column of classes, call `write_predictions`, providing it with the name of the CSV file to be read and updated.

```
>>>iris_forest.write_predictions('iris_predictions.csv')
Classified 9 individuals.
```

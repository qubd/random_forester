random_forester
===============

Minimal random forest classifier in Python.

Run forester.py to do classification of quantitative data. Everything is written from scratch, so no scientific packages are required. The training data given here is Fisher's iris flower data set, and the prediction data is a sample of nine individuals from that data set, three of each species, with the entry in the species column removed.

Each of the two data files must consist of comma separated values. Variable names go in the first row, and class names in the last column of the training data. All variables must be quantitative, save for the class names, which should not be present in the prediction data set (or why would you need to do prediction?).

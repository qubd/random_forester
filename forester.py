#Random Forest Classifier
#Brendan Cordy, 2014

from random import randrange
from random import shuffle
from operator import itemgetter
from math import log

#Datasets------------------------------------------------------------------------------------------------

class DataCSV(object):
    def __init__(self, str_data, vrs, cls, ind):
        self.str_data = str_data
        self.variables = vrs
        self.classes = cls
        self.indivs = ind

    #Input data with classifications given, to be used for training.
    @classmethod
    def from_training_data(cls, filename):
        with open(filename,'r') as input_file:
            input_data = [line.rstrip('\n').rstrip('\r').split(',') for line in input_file]

        #Ignore empty rows (an extra newline at the end of the file will trigger this).
        no_empty_rows = [x for x in input_data if x != ['']]
        str_data = no_empty_rows

        #Extract variable names from the top row.
        variables = str_data[0]
        #Convert strings representing values of quantitative variables to floats. The last entry in each
        #row will be the class name, so don't attempt to convert that to a float.
        indivs = [[float(val) for val in line[:-1]]+[line[-1]] for line in str_data[1:]]
        #Extract class names and remove duplicates.
        classes = list(set([row[-1] for row in indivs]))

        print "Data for " + str(len(indivs)) + " individuals with " + str(len(variables)-1) + " quantitative variables."
        print "Classes: " + str(classes)

        return cls(str_data, variables, classes, indivs)

    #Input data without classifications given. A list of possible classes must be provided.
    @classmethod
    def from_prediction_data(cls, filename, classes):
        with open(filename,'r') as input_file:
            input_data = [line.rstrip('\n').rstrip('\r').split(',') for line in input_file]

        #Ignore empty rows (an extra newline at the end of the file will trigger this).
        no_empty_rows = [x for x in input_data if x != ['']]
        str_data = no_empty_rows

        #Extract variable names from the top row.
        variables = str_data[0]
        #Convert strings representing values of quantitative variables to floats.
        indivs = [[float(val) for val in line] for line in str_data[1:]]

        return cls(str_data, variables, classes, indivs)

    #Return a random sample (with replacement) of n individuals from the data set.
    def to_sample(self):
        return Sample(self.indivs, self.variables)

    #Return a random sample (with replacement) of n individuals from the data set.
    def random_sample(self, n):
        sample_data = []
        for i in range(0, n):
            j = randrange(0, len(self.indivs))
            sample_data.append(self.indivs[j])
        return Sample(sample_data, self.variables)

    #Randomly reorder the individuals in the data set.
    def randomize_order(self):
        shuffle(self.indivs)

    #Partition in the individuals into k more or less equally sized disjoint subsets. Return the ith such subset
    #as a new dataset, and its complement as another. Used for cross-validation.
    def validation_partition(self, i, k):
        first_row = (i - 1) * (len(self.indivs) / k)
        #If this is the last fold (last subset), make sure to include all the data.
        if i == k:
            last_row = len(self.indivs)
        else:
            last_row = (i) * (len(self.indivs) / k)

        valid_data = DataCSV(
            self.str_data[0] + self.str_data[first_row:last_row],
            self.variables,
            self.classes,
            self.indivs[first_row:last_row]
        )
        train_data = DataCSV(
            self.str_data[0] + self.str_data[0:first_row] + self.str_data[last_row:len(self.str_data)],
            self.variables,
            self.classes,
            self.indivs[0:first_row] + self.indivs[last_row:len(self.indivs)]
        )

        return train_data, valid_data

    #Write the data as a csv file.
    def write(self, filename):
        lines = []
        variables_row = (','.join(self.variables))
        lines.append(variables_row)
        #Convert all data values to strings and build list of lines.
        for indiv in self.indivs:
            for index,value in enumerate(indiv):
                indiv[index] = str(value)
            indiv_row = (','.join(indiv))
            lines.append(indiv_row)

        with open(filename, 'w') as out_file:
            out_file.write('\n'.join(lines))

#Samples-------------------------------------------------------------------------------------------------

class Sample(object):
    #Collect classes present in the sample by extracting each individual's class, then removing duplicates.
    def __init__(self, sample_indivs, variables):
        self.indivs = sample_indivs
        self.variables = variables
        self.classes = list(set([x[-1] for x in sample_indivs]))

    #Calculate the split with the largest information gain, and return the two samples obtained by splitting.
    def find_best_split(self):
        best_var, best_index_all, best_info_gain_all = 0, 0, 0
        #Loop over all non-class variables (columns) in the data set, to find the best one to split on.
        for column in range(0, len(self.variables) - 1):
            #Sort the sample by that variable
            self.indivs.sort(key=itemgetter(column))
            best_split_index, best_info_gain = 0, 0
            #Evaluate every possible splitting index.
            for i in range(1, len(self.indivs)):
                info_gain = self.eval_split(i)
                if info_gain > best_info_gain:
                    best_split_index, best_info_gain = i, info_gain
            #If this variable's best split is the best of all splits yet observed, keep it.
            if best_info_gain > best_info_gain_all:
                best_var, best_index_all, bext_info_gain_all = column, best_split_index, best_info_gain

        #Return the best variable and the value to split on, as well as the two halves after the split. Note
        #that the split vales are rounded to four decimal places. In data sets with variables that range over
        #a very small set of values this should be changed.
        self.indivs.sort(key=itemgetter(best_var))
        #The split value is the median of the values in the two individuals closest to the split.
        split_value = round(0.5 * (self.indivs[best_index_all][best_var] + self.indivs[best_index_all - 1][best_var]), 4)
        left_side_sample = Sample(self.indivs[:best_index_all], self.variables)
        right_side_sample = Sample(self.indivs[best_index_all:], self.variables)

        return best_var, split_value, left_side_sample, right_side_sample

    #Evaluate split using information gain: Compute the entropy of that class distribution in the sample,
    #and compute the weighted sum of the entropies of the two resulting class distributions on each side of
    #the split. A higher difference mean more information gain, and a better split.
    def eval_split(self, index):
        counts_total, counts_left, counts_right = [], [], []
        #Compute proportion of individuals on each side of the split (assume the data is already sorted).
        prop_left = index/float(len(self.indivs))
        prop_right = (len(self.indivs) - index)/float(len(self.indivs))
        #Tally up the counts of classes prior to splitting, and the counts on each side of the split.
        for i in range(0, len(self.classes)):
            counts_total.append(sum(indiv.count(self.classes[i]) for indiv in self.indivs))
            counts_left.append(sum(indiv.count(self.classes[i]) for indiv in self.indivs[:index]))
            counts_right.append(sum(indiv.count(self.classes[i]) for indiv in self.indivs[index:]))
        #Calculate entropies and return information gain.
        return entropy(counts_total) - ((prop_left * entropy(counts_left)) + (prop_right * entropy(counts_right)))

#DecisionTree--------------------------------------------------------------------------------------------

class DecisionTree(object):
    def __init__(self, sample, depth):
        #If there is only one class in the sample, this node is a leaf labelled with that class.
        if len(sample.classes) == 1:
            self.leaf = True
            self.cls = sample.classes[0]
        #If this node is at maximum depth, it's a leaf labelled with the most common class in the sample.
        elif depth == 0:
            self.leaf = True
            self.cls = max(sample.classes, key=sample.classes.count)
        #Otherwise, find the best split and create two new subtrees.
        else:
            self.leaf = False
            self.split_var, self.split_val, left_sample, right_sample = sample.find_best_split()
            self.left = DecisionTree(left_sample, depth - 1)
            self.right = DecisionTree(right_sample, depth - 1)

    def classify(self, indiv):
        if self.leaf:
            return self.cls
        else:
            if indiv[self.split_var] <= self.split_val:
                return self.left.classify(indiv)
            else:
                return self.right.classify(indiv)

#RandomForest--------------------------------------------------------------------------------------------

class RandomForest(object):
    #Construct a forest of decision trees built with random samples of a given size from a given dataset.
    def __init__(self, data, num_trees, max_depth, sample_size):
        self.data = data
        self.classes = data.classes
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []
        for i in range(num_trees):
            sample = data.random_sample(sample_size)
            self.trees.append(DecisionTree(sample, max_depth))

    def classify(self, indiv):
        #Create a dictionary to tally up the votes of each of the decision trees.
        votes = {x : 0 for x in self.classes}
        for tree in self.trees:
            votes[tree.classify(indiv)] += 1
        #Find the class with the largest number of votes.
        winning_class = max(votes.iteritems(), key=itemgetter(1))[0]
        return winning_class

    def cross_validate(self, k):
        self.data.randomize_order()
        correct_class_rates = []
        #Partition individuals in the dataset into disjoint subsets of size k.
        for i in range(1, k + 1):
            train_subset, valid_subset = self.data.validation_partition(i,k)
            #Create a random forest with the larger training portion of the partition.
            subset_forest = RandomForest(train_subset, self.num_trees, self.max_depth, self.sample_size)

            #Classify the individuals in the validation part of the partition using the new forest.
            count_cor, count_inc = 0, 0
            for indiv in valid_subset.indivs:
                #Check if the predicted class matches.
                if subset_forest.classify(indiv) == indiv[-1]:
                    count_cor += 1
                else:
                    count_inc += 1

            total = count_cor + count_inc
            correct_class_rates.append(100 * (float(count_cor) / total))
            print "Fold " + str(i) + ": " + str(count_cor) + " of " + str(total) + " individuals classified correctly."

        avg_correct_rate = sum(correct_class_rates) / float(len(correct_class_rates))
        print "Overall success rate: " + str(round(avg_correct_rate, 1)) + "%"

    def write_predictions(self, filename):
        prediction_data = DataCSV.from_prediction_data(filename, self.classes)
        #Add the class as a new variable in the prediction data.
        prediction_data.variables.append(self.data.variables[-1])
        #Add the classification for each individual.
        for indiv in prediction_data.indivs:
            indiv.append(self.classify(indiv))

        prediction_data.write(filename)

#Top-Level Helper Functions------------------------------------------------------------------------------

#Compute the entropy of a list of frequencies.
def entropy(freq_list):
    #Scale the list so its sum is one.
    prob_list = [x / float(sum(freq_list)) for x in freq_list]
    ent = 0
    #Compute each term in the sum which defines information entropy. We need to deal with the indeterminate
    #form which arises when one of the frequencies is zero separately to avoid a domain issue.
    for x in prob_list:
        if x > 0:
            ent += -x * log(x,2)
        elif x == 0:
            ent += float(0)
    return ent

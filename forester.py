#Minimalist Random Forest Classifier
#Brendan Cordy, 2014

from random import randrange
from operator import itemgetter
import math

#INPUT--------------------------------------------------------------

#Data files are csv. First row consists of variable names, all variables are quantitative
#except for the class names, which must be in the last column. Extra empty lines and extra
#commas at the end of rows are not a problem.
def parse_csv(filename):
    with open(filename,'r') as file:
        parsed_data = [line.rstrip('\n').split(',') for line in file]
        fix_empty_rows = [x for x in parsed_data if x != ['']]
        fix_extra_commas = [[x for x in y if x != ''] for y in fix_empty_rows]
        return fix_extra_commas

#Convert numerical data entries to floats, in place.
def floatify(rows_of_data):
    for r in rows_of_data:
        for i,c in enumerate(r):
           if is_number(c):
               r[i] = float(c)

def is_number(s):
    pieces = s.split('.')
    if len(pieces) > 2:
        return False
    else:
        return all([x.isdigit() for x in pieces])

#SAMPLING------------------------------------------------------------

def random_sample_with_replacement(n):
    sample = []
    for i in range(0,n):
        j=randrange(0,len(training_data))
        sample.append(training_data[j])
    return sample

#SPLITTING------------------------------------------------------------

def find_best_split(sample):
    #If the sample contains only one class, return that class.
    classes_in_sample = list(set([x[len(sample[0]) - 1] for x in sample]))
    if len(classes_in_sample) == 1:
        #The -1 in the first entry below is a flag that there is only one class in the sample,
        #and that class is returned as the first element of the list in the third slot.
        return (-1, 0, [classes_in_sample[0]], [])
    #Otherwise, we need to search for the best split.
    else:
        best_var = 0
        best_index_allvars = 0
        best_info_gain_allvars = 0
        #Loop over all non-class variables in the data set.
        for variable in range(0, len(names) - 1):
            #Sort the sample by that variable
            sorted_sample = sorted(sample, key=itemgetter(variable))
            best_split_index = 0
            best_info_gain = 0
            #Evaluate every possible splitting index and keep the best.
            for index,individual in enumerate(sorted_sample):
                #Individuals whose data values are < index end up on the left side of the split
                #so splitting at the first individual is not splitting at all.
                if(index != 0):
                    info_gain = eval_split(sorted_sample, index)
                    if(info_gain > best_info_gain):
                        best_split_index = index
                        best_info_gain = info_gain
            #Hold on to the best split, both the variable and split index.
            if(best_info_gain > best_info_gain_allvars):
                best_var = variable
                best_index_allvars = best_split_index
                best_info_gain_allvars = best_info_gain
        #Return the best variable and the value to split on, as well as the two halves after the split.
        #Note that the split vales are rounded to four decimal places to make the output pretty. In
        #applications where variables may take very small values this should be changed.
        sorted_sample = sorted(sample, key=itemgetter(best_var))
        split_value = round(0.5 * (sorted_sample[best_index_allvars][best_var] + sorted_sample[best_index_allvars - 1][best_var]),4)
        return (best_var, split_value, sorted_sample[:best_index_allvars], sorted_sample[best_index_allvars:])

def eval_split(sorted_sample, index):
    #Use information gain: Compute the entropy of that class distribution in the sample, and compute the 
    #weighted sum of the entropies of the two resulting class distributions on each side of the split. 
    #A higher difference mean more information gain, and a better split. For GINI impurity based splitting
    #see older versions.
    counts_total = []
    counts_left = []
    counts_right = []
    #Compute proportion of individuals on each side of the split.
    proportion_left = index/float(len(sorted_sample))
    proportion_right = (len(sorted_sample) - index)/float(len(sorted_sample))
    #Tally up the counts of classes prior to splitting, and on each side of the split.
    for i in range(0, len(classes)):
        counts_total.append(sum(Indiv.count(classes[i]) for Indiv in sorted_sample))
        counts_left.append(sum(Indiv.count(classes[i]) for Indiv in sorted_sample[:index]))
        counts_right.append(sum(Indiv.count(classes[i]) for Indiv in sorted_sample[index:]))
    #Calculate entropies and information gain.
    return entropy(counts_total)-((proportion_left*entropy(counts_left))+(proportion_right*entropy(counts_right)))

def entropy(freq_list):
    #Scale the list so its sum is one.
    prob_list = [x/float(sum(freq_list)) for x in freq_list]
    #Compute each term in the sum which defines information entropy. We need to deal with the indeterminate
    #form which arises when one of the frequencies is zero separately to avoid a domain issue.
    e_list = []
    for x in prob_list:
        if x > 0:
            e_list.append(-x*math.log(x,2))
        elif x == 0:
            e_list.append(float(0))
    return sum(e_list)

#DECISION TREE-------------------------------------------------------

def build_decision_tree(sample, height):
    #Each node has the form [var,split_val,left,right] where left and right are nodes.
    #or classes.
    (split_var, split_val, left_side, right_side) = find_best_split(sample)
    #If there was only one class in the sample, return that class. Recall that find_best_split will
    #detect this and return that class as the first element of the list left_side.
    if split_var == -1:
        return left_side[0]
    #If this node is a leaf, return the most common class in the sample.
    elif height == 0:
        strip_classes = list(set([x[len(sample[1]) - 1] for x in sample]))
        return max(set(strip_classes), key=strip_classes.count)
    else:
        return [split_var, split_val, build_decision_tree(left_side, height - 1), build_decision_tree(right_side, height - 1)]

def classify_individual(indiv, tree):
    #If the current node is a leaf (class name), return that leaf.
    if isinstance(tree, basestring):
        return tree
    #If not, check if the split variable is less than or equal to the split value and move down the
    #tree in the right direction.
    else:
        if indiv[tree[0]] <= tree[1]:
            return classify_individual(indiv, tree[2])
        else:
            return classify_individual(indiv, tree[3])

#RANDOM FOREST-------------------------------------------------------

def build_forest(forest_size, height, sample_size):
    forest = []
    for i in range(0, forest_size):
        forest.append(build_decision_tree(random_sample_with_replacement(sample_size), height))
    return forest

def classify_with_forest(indiv, forest):
    votes = []
    #Initialize the list of votes.
    for i in range(0, len(classes)):
        votes.append(0)
    #Find the class each tree votes for and increment the list of votes in the right index.
    for j in range(0, len(forest)):
        tree_vote = classify_individual(indiv, forest[j])
        for k in range(0,len(classes)):
            if(tree_vote == classes[k]):
                votes[k] += 1
    winner_index = votes.index(max(votes))
    return classes[winner_index]

def validate_forest(forest):
    #Keep track of correct and incorrect classifications.
    counts = [0,0]
    for indiv in validation_data:
        #Check if the predicted class matches.
        if(classify_with_forest(indiv,forest) == indiv[len(indiv) - 1]):
            counts[0] += 1
        else:
            counts[1] += 1
    return 100*(float(counts[0]) / (len(validation_data)))
           

#PREDICTION----------------------------------------------------------

def make_predictions(forest):
    #Classify each individual in the prediction data.
    for indiv in prediction_data:
        indiv.append(classify_with_forest(indiv, forest))
    #Write the classified data to the prediction data file.
    with open('prediction.csv', 'w') as pred_file:
        names_row = (','.join(names))
        pred_file.write(names_row)
        #Convert prediction data to strings and write row by row.
        for indiv in prediction_data:
            pred_file.write('\n')
            for i,attrib in enumerate(indiv):
                indiv[i] = str(attrib)
            indiv_row = (','.join(indiv))
            pred_file.write(indiv_row)
    
#EXECUTION-----------------------------------------------------------

#Parse the training, testing, and prediction csv files.
raw_training_data = parse_csv('training.csv')
raw_validation_data = parse_csv('validation.csv')
raw_prediction_data = parse_csv('prediction.csv')

#Strip variable names and convert numerical values to floats.
#Everybody loves global variables, right?
training_data = raw_training_data[1:]
floatify(training_data)
validation_data = raw_validation_data[1:]
floatify(validation_data)
prediction_data = raw_prediction_data[1:]
floatify(prediction_data)

#Extract variable names and class names from the training data.
names = raw_training_data[0]
classes = list(set([x[len(training_data[1]) - 1] for x in training_data]))

#Echo the training data and class names as a sanity check.
print "Training Data:"
print training_data
print "\n"
print "Classes: "
print classes
print "\n"
print "Forest: "

#Build and validate the random forest. The three parameters are:
#1 - The number of decision trees in the forest.
#2 - The maximum depth of the trees in the forest.
#3 - The size of the random sample used to build each decision tree.
rf = build_forest(100, 3, 30)
print rf
print "\n"
print "Percentage of Validation Data Classifed Correctly: "
print validate_forest(rf)

#Classify the prediction data, and write the results to the prediction data file.
#Note that running the program multiple times on the same prediction data will
#append the new classifications instead of overwriting previous results.
make_predictions(rf)

import bag_of_words
import evaluation_metrics
from decimal import Decimal
from math import log10 as log
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



def test_multinomial_naive_bayes(prior, conditional_probability, conditional_probability_of_non_occurring_word,
                                 an_email_bag_of_words_test):
    """

    conditional_probability_of_non_occurring_word: This is the conditional probability for each word in the testing set which is not in the training data
    prior: This is the prior for all classes
    conditional_probability:  This is the conditional probability for each word in vocabulary in spam and ham data
    an_email_bag_of_words_test: This is the given test instance we want to classify
    the class of the given email
    """
    score = {}
    for each_class in list(prior):
        score[each_class] = prior[each_class]
        for word in list(an_email_bag_of_words_test):
            if an_email_bag_of_words_test[word] != 0:
                try:
                    score[each_class] += conditional_probability[each_class][word]
                # This is the case if the word was not in the train data and thus the laplace pruning gives this result
                except KeyError:
                    score[each_class] += conditional_probability_of_non_occurring_word[each_class]
    # Here we are taking spam as 1 and ham as -1
    if score["spam"] > score["ham"]:
        return 1
    else:
        return 0

#taking dataset path as input

dataset_name=input("Enter folder path: ")
train_data_set=dataset_name+"/train"
test_data_set=dataset_name+"/test"

#converting train dataset into bag of words
spam_bag_of_words,ham_bag_of_words,all_words_freq,freq_of_all_words_in_spam,freq_of_all_words_in_ham,total_no_of_files,no_of_spam_files, no_of_ham_files, all_words_dict=bag_of_words.bag_of_words(train_data_set)


prior = {}
conditional_probability = {}
conditional_probability["spam"] = {}
conditional_probability["ham"] = {}
conditional_probability_of_non_occurring_word = {}
conditional_probability_of_non_occurring_word["spam"] = {}
conditional_probability_of_non_occurring_word["ham"] = {}
value = Decimal(no_of_spam_files / float(total_no_of_files))
# First we calculate the priors for the spam and ham dataset
prior["spam"] = log(value)

total_no_of_words_in_ham = sum(freq_of_all_words_in_ham.values())
prior["ham"] = log(no_of_ham_files / float(total_no_of_files))
total_no_of_words_in_spam = sum(freq_of_all_words_in_spam.values())
# Now we calculate the values for the conditional probabilities
for word in list(freq_of_all_words_in_spam):
    conditional_probability["spam"][word] = log((freq_of_all_words_in_spam[word] + 1) / (
        float(total_no_of_words_in_spam + len(all_words_freq))))

# Now we will do the same procedure for ham docs
for word in list(freq_of_all_words_in_ham):
    conditional_probability["ham"][word] = log((freq_of_all_words_in_ham[word] + 1) / (
        float(total_no_of_words_in_ham + len(all_words_freq))))
# These are the values for the conditional probabilities whose words are not in the training dataset
conditional_probability_of_non_occurring_word["ham"] = log(
    1 / (float(total_no_of_words_in_ham + len(all_words_freq))))
conditional_probability_of_non_occurring_word["spam"] = log(
    1 / (float(total_no_of_words_in_spam + len(all_words_freq))))

spam_bag_of_words,ham_bag_of_words,all_words_freq,freq_of_all_words_in_spam,freq_of_all_words_in_ham,total_no_of_files,no_of_spam_files, no_of_ham_files, all_words_dict=bag_of_words.bag_of_words(test_data_set)


spam_predict = []
for each_document in spam_bag_of_words:
    spam_predict.append(test_multinomial_naive_bayes(prior, conditional_probability,
                                                                                conditional_probability_of_non_occurring_word,
                                                                                each_document))
# We  are taking spam as 1
spam_actual = [1] * len(spam_predict)
ham_predict = []
for each_document in ham_bag_of_words:
    ham_predict.append(test_multinomial_naive_bayes(prior, conditional_probability,
                                                                                conditional_probability_of_non_occurring_word,
                                                                                each_document))
ham_actual = [0] * len(ham_predict)
total_actual = spam_actual + ham_actual
total_predict = spam_predict + ham_predict
# Now we find the evaluation metrics for the method
accuracy = evaluation_metrics.accuracy(total_actual, total_predict)
precision = evaluation_metrics.precision(total_actual, total_predict)
recall = evaluation_metrics.recall(total_actual, total_predict)
f1_score = evaluation_metrics.f1_score(recall, precision)
print("The accuracy is", accuracy)
print("The Precision is", precision)
print("The Recall is", recall)
print("The F1 Score is", f1_score)
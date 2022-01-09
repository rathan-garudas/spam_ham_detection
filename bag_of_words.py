import copy
import glob
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#taking dataset path as a input
def bag_of_words(path):

    ham_path=path+"/ham"
    spam_path=path+"/spam"

    #taking list of all txt files in the folder
    spam_filenames = glob.glob(spam_path + "/" + "*.txt")
    ham_filenames = glob.glob(ham_path + "/" + "*.txt")

    spam_content=[]
    ham_content=[]
    total_content=""
    
    #reading the content in files and storing them
    for files in spam_filenames:
        temp=open(files, "r", encoding='utf-8', errors='ignore').read()
        spam_content.append(temp)
        total_content = total_content + " " + temp
    
    for files in ham_filenames:
        temp=open(files, "r", encoding='utf-8', errors='ignore').read()
        ham_content.append(temp)
        total_content = total_content + " " + temp
    
    
    
    no_of_ham_files=len(ham_filenames)
    no_of_spam_files=len(spam_filenames)
    total_no_of_files=len(spam_filenames)+len(ham_filenames)

    all_words_dict = {}
    #breaking the text into words
    all_words = re.findall("[a-zA-Z]+", total_content)
    all_words_freq = {}
    # these are the words which doesn't contribute to the model
    stopwords=["a", "about", "above", "across", "after", "afterwards", "again", "all", "almost", "alone", 
                                "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst",
                                "amount", "an", "and","another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", 
                                "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", 
                                "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot",
                                "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", 
                                "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", 
                                "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", 
                                "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", 
                                "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", 
                                "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", 
                                "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", 
                                "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", 
                                "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", 
                                "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", 
                                "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", 
                                "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", 
                                "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", 
                                "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they","this", "those", "though", 
                                "through", "throughout","thru", "thus", "to", "together", "too", "toward", "towards","under", "until", 
                                "up", "upon", "us","very", "was", "we", "well", "were", "what", "whatever", "when","whence", "whenever", 
                                "where", "whereafter", "whereas", "whereby","wherein", "whereupon", "wherever", "whether", "which", 
                                "while", "who", "whoever", "whom", "whose", "why", "will", "with","within", "without", "would", "yet", 
                                "you", "your", "yours", "yourself", "yourselves"]
    for word in all_words:
        word = word.lower()
        if word not in all_words_dict:
            if word not in stopwords:
                all_words_dict[word] = 0
        if word in all_words_freq:
            if word not in stopwords:
                all_words_freq[word] = all_words_freq[word] + 1
        else:
            if word not in stopwords:
                all_words_freq[word] = 1
    # In the following steps we find the words in the spam dataset and create the bag of words
    spam_bag_of_words = []
    freq_of_all_words_in_spam = {}
    # The frequencies of words are stored in the case of bag of words.
    for each_spam_mail in spam_content:
        temp_dict = copy.deepcopy(all_words_dict)
        words_in_each_spam_mail = re.findall("[a-zA-Z]+", each_spam_mail)
        # Here we create the bag of words for each document and append it in a list
        for word in words_in_each_spam_mail:
            word = word.lower()
            if word in temp_dict:
                temp_dict[word] = temp_dict[word] + 1
        # Here we store all the words in the spam dataset
        freq_of_all_words_in_spam = Counter(freq_of_all_words_in_spam) + Counter(temp_dict)
        spam_bag_of_words.append(temp_dict)
    # In the following steps we find all the ham words and add them in the bag of words
    freq_of_all_words_in_ham = {}
    ham_bag_of_words = []
    for each_ham_mail in ham_content:
        # Here we create the bag of words for each document and append it in a list
        temp_dict = copy.deepcopy(all_words_dict)
        words_in_each_ham_mail = re.findall("[a-zA-Z]+", each_ham_mail)
        for word in words_in_each_ham_mail:
            word = word.lower()
            if word in temp_dict:
                temp_dict[word] = temp_dict[word] + 1
        # Here we store all the words in the ham dataset
        freq_of_all_words_in_ham = Counter(freq_of_all_words_in_ham) + Counter(temp_dict)
        ham_bag_of_words.append(temp_dict)
    return spam_bag_of_words, ham_bag_of_words, all_words_freq, freq_of_all_words_in_spam, freq_of_all_words_in_ham, total_no_of_files, no_of_spam_files, no_of_ham_files, all_words_dict


""" 
    spam_bag_of_words:  This is a list of list all words in each spam email
    ham_bag_of_words: This is a list of list all words in each ham email
    all_words_freq: This is a dicitionary containing all words in the dataset and their frequencies
    freq_of_all_words_in_spam: This is a dicitionary containing all words in the spam emails and their frequencies
    freq_of_all_words_in_ham: This is a dicitionary containing all words in the ham emails and their frequencies
    total_no_of_files: This is total number of emails in all dataset
    no_of_spam_files: This is total number of spam emails in all spam dataset
    no_of_ham_files: This is total number of ham emails in all ham dataset
    all_words_dict: This is a list containing all the words in the training dataset"""
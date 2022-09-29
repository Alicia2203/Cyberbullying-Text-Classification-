#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:50:46 2022

@author: alicia
"""
###############################################################################
###############################################################################
                   # PART 1: SCRAPE DATA FROM TWITTER  #
###############################################################################
###############################################################################

# import packages
import tweepy
import pandas as pd # For saving the response data in CSV format

###############################################################################
               # Get Authenticaiton Acess from Twitter Api  #
###############################################################################

consumer_key = "xxx"
consumer_secret = "xxx" 
access_key = "xxx"
access_secret = "xxx" 

def twitter_setup():
    # Authentication and access using keys
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    # Calling API
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    try:
        api.verify_credentials()
        print("Successful Authentication")
    except:
        print("Failed authentication")
    return api
    
# An extractor object to hold the api data by calling in twitter_setup()function
extractor = twitter_setup()

###############################################################################
                          # Scrape Random Tweets  #
###############################################################################

# create a function that Extract tweets based on keywords
def keywords_tweets(api, keyword, number_of_tweets):
    new_keyword = keyword + " -filter:retweets"
    tweets = []
    for status in tweepy.Cursor(api.search_tweets, q=new_keyword, 
                                lang="en", tweet_mode='extended', 
                                result_type='mixed').items(number_of_tweets):
          tweets.append(status)
    return tweets

potent_random_tweets = keywords_tweets(extractor, "a OR b OR c OR d OR e OR f OR g OR h OR i OR j OR k OR l OR m OR n OR o OR p OR q OR r OR s OR t OR u OR v OR w OR x OR y OR z", 
                                    1000)

# create a pandas DataFrame by looping through each element and add it to the DataFrame
random_tweets = pd.DataFrame(data=[tweet.id for tweet in potent_random_tweets], 
                    columns=['Tweets_ID'])
random_tweets['label'] = 0 # defaulted as 0 before manually relabelling each tweet
random_tweets['tweet'] = [tweet.full_text for tweet in potent_random_tweets]

###############################################################################
                  # Scrape tweets with search term "nigger"  #
###############################################################################
nigger_tweets = keywords_tweets(extractor, "nigger",
                                    100)

# create a pandas DataFrame by looping through each element and add it to the DataFrame
nigger_hate_tweets = pd.DataFrame(data=[tweet.id for tweet in nigger_tweets], 
                    columns=['Tweets_ID'])
nigger_hate_tweets['label'] = 1 # defaulted as 1 before manually relabelling each tweet
nigger_hate_tweets['tweet'] = [tweet.full_text for tweet in nigger_tweets]

###############################################################################
          # Scrape tweets with search term "whitetrash OR whitepeople"  #
###############################################################################

whitetrash_tweets = keywords_tweets(extractor, "whitetrash OR whitepeople",
                                    100)

# create a pandas DataFrame by looping through each element and add it to the DataFrame
whitetrash_hate_tweets = pd.DataFrame(data=[tweet.id for tweet in whitetrash_tweets], 
                    columns=['Tweets_ID'])
whitetrash_hate_tweets['label'] = 1 # defaulted as 1 before manually relabelling each tweet
whitetrash_hate_tweets['tweet'] = [tweet.full_text for tweet in whitetrash_tweets]

###############################################################################
                  # Scrape tweets with search term "bitch"  #
###############################################################################

bitch_tweets = keywords_tweets(extractor, "bitch",
                                    100)

# create a pandas DataFrame by looping through each element and add it to the DataFrame
bitch_hate_tweets = pd.DataFrame(data=[tweet.id for tweet in bitch_tweets], 
                    columns=['Tweets_ID'])
bitch_hate_tweets['label'] = 1 # defaulted as 1 before manually relabelling each tweet
bitch_hate_tweets['tweet'] = [tweet.full_text for tweet in bitch_tweets]

###############################################################################
                  # Scrape tweets with search term "gay"  #
###############################################################################

gay_tweets = keywords_tweets(extractor, "gay",
                                    100)

# create a pandas DataFrame by looping through each element and add it to the DataFrame
gay_hate_tweets = pd.DataFrame(data=[tweet.id for tweet in gay_tweets], 
                    columns=['Tweets_ID'])
gay_hate_tweets['label'] = 1 # defaulted as 1 before manually relabelling each tweet
gay_hate_tweets['tweet'] = [tweet.full_text for tweet in gay_tweets]


###############################################################################
                          # Combine and Export data  #
###############################################################################

# combine all seperate data frames into one
frames = [random_tweets, nigger_hate_tweets, whitetrash_hate_tweets, bitch_hate_tweets, gay_hate_tweets]
data = pd.concat(frames, ignore_index=True)


# import to excel file for manual relabelling purposes
data.to_excel('/Users/alicia/Library/CloudStorage/OneDrive-SunwayEducationGroup/Sunway/Sem 6/SMA/Group Assignment 2/train_test_data.xlsx')

# read labelled data set
labelled_data = pd.read_excel('train_test_data.xlsx')

# import to labelled data set csv file 
labelled_data.to_csv('/Users/alicia/Library/CloudStorage/OneDrive-SunwayEducationGroup/Sunway/Sem 6/SMA/Group Assignment 2/train_test_data.csv')


###############################################################################
###############################################################################
                   # PART 2: CREATE CLASSIFICAITON MODEL  #
###############################################################################
###############################################################################

# Import Packages
import pandas as pd # data manipulation
import numpy as np # mathematical calculation
import re # regular expression
import matplotlib.pyplot as plt # data visualization
import seaborn as sns
from wordcloud import WordCloud 
import string
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# import dataset that has been manually labelled
train_test_tweets = pd.read_csv('train_test_data.csv',
                                  usecols= [1,2,3])

###############################################################################
                             # Explore Data  #
###############################################################################

# Explore Data 
train_test_tweets.head()

train_test_tweets.info()

# check if there are any missing values
train_test_tweets.isnull().sum()

sum(train_test_tweets["label"] == 0)
sum(train_test_tweets["label"] == 1)

# Create bar chart to visualize number of hate and non-hate tweets
fig = plt.figure(figsize=(5,5))
ax = sns.countplot(x='label', data = train_test_tweets)
abs_values = train_test_tweets['label'].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=abs_values)


###############################################################################
                             # Data Cleaning  #
###############################################################################

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt   


# remove twitter handles (@user)
train_test_tweets['cleaned_tweet'] = np.vectorize(remove_pattern)(train_test_tweets['tweet'],"@[\w]*")

# remove hyperlinks
train_test_tweets['cleaned_tweet']= train_test_tweets['cleaned_tweet'].str.replace(r"http\S+", " ")

# remove special characters, numbers, punctuations
train_test_tweets['cleaned_tweet']= train_test_tweets['cleaned_tweet'].str.replace("[^a-zA-Z]", " ")

# Tokenization
tweets_token = train_test_tweets['cleaned_tweet'].apply(word_tokenize).tolist()


# Create stop words list
stop_words = stopwords.words('english')

# Create remove noise function
def remove_noise(tweet_tokens, stop_words):
    cleaned_tokens = []
    for token in tweet_tokens:
        if (len(token) > 3) and (token not in string.punctuation) and (token.lower() not in stop_words):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# Clean tokens
cleaned_tokens = []
for tokens in tweets_token:
    rm_noise = remove_noise(tokens,stop_words)
    lemma_tokens = lemmatize_sentence(rm_noise)
    cleaned_tokens.append(lemma_tokens)

# Convert lists of tokens into one list with all tokens
def get_all_words(cleaned_tokens_list):
    tokens=[]
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# Convert lists of tokens into one list with all tokens with get_all_words function
tokens_flat = get_all_words(cleaned_tokens)

# Join back the cleaned tokens into sentences
new_tweet = []
for line in cleaned_tokens:
    line = ' '.join(line)
    new_tweet.append(line)

###############################################################################
                 # Visualize most frequent Token Count #
###############################################################################

# Creating FreqDist, keeping the 20 most common tokens
freq_dist = FreqDist(tokens_flat)
all_fdist = freq_dist.most_common(20)

# Conversion to Pandas series via Python Dictionary for easier plotting
all_fdist = pd.Series(dict(all_fdist))

# Setting figure, ax into variables
fig, ax = plt.subplots(figsize=(10,6))

# Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
plt.xticks(rotation=30);


###############################################################################
                               # Word Cloud #
###############################################################################
non_hate_tweets = train_test_tweets[train_test_tweets.label == 0]
hate_tweets = train_test_tweets[train_test_tweets.label == 1]

train_test_tweets['cleaned_tweet'] = train_test_tweets['cleaned_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
train_test_tweets['cleaned_tweet'] = train_test_tweets['cleaned_tweet'].apply(lambda x: " ".join(x for x in x.split() if len(x) > 3))

non_hate_tweets = " ".join(train_test_tweets['cleaned_tweet'][train_test_tweets.label == 0])
hate_tweets = " ".join(train_test_tweets['cleaned_tweet'][train_test_tweets.label == 1])

##################### Wordcloud for non_hate_tweets ###########################
wordcloud = WordCloud(width = 1600, height = 800,collocations = False,
                background_color ='white',
                min_font_size = 10).generate(non_hate_tweets)

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

######################### Wordcloud for hate_tweets ###########################
wordcloud = WordCloud(width = 1600, height = 800,
                background_color ='black',
                min_font_size = 10).generate(hate_tweets)

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

###############################################################################
          # Feature Extraction - Vectorize Tweets with TF-IDF #
###############################################################################

tf = TfidfVectorizer(max_features=1000)
X = tf.fit_transform(new_tweet).toarray()

###############################################################################
                                # Model Building  #
###############################################################################

# assign predictor
Y = train_test_tweets['label']

# split data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=(0))

print("Size of x_train:", (X_train.shape))
print("Size of y_train:", (Y_train.shape))
print("Size of x_test: ", (X_test.shape))
print("Size of y_test: ", (Y_test.shape))


###############################################################################
                          # Logistic Regression model  #
###############################################################################

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
logreg_predict = logreg.predict(X_test)
logreg_acc = accuracy_score(logreg_predict, Y_test)

print("\n")
print("+----------------------------------------------------+")
print(" Logistic Regression model Classification report:")
print("+----------------------------------------------------+")
print(classification_report(Y_test, logreg_predict))
print("Test accuarcy: {:.2f}%".format(logreg_acc*100))

#Get the confusion matrix.
print("\n")
print("+----------------------------------------------------+")
print(" Logistic Regression model Confusion Matrix:")
print("+----------------------------------------------------+")
print(confusion_matrix(Y_test, logreg_predict))
print("\n")

cm = confusion_matrix(Y_test, logreg_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

###############################################################################
                   # Multinomial Naive Bayes model  #
###############################################################################

# Fit Multinomial Naive Bayes model
MNB_model = MultinomialNB()                                                    
MNB_model.fit(X_train, Y_train)
MNB_Y_pred = MNB_model.predict(X_test)
MNB_acc = accuracy_score(MNB_Y_pred, Y_test)

# Create classification report
MNB_cf = classification_report(Y_test, MNB_Y_pred) 
print("\n")
print("+----------------------------------------------------+")
print(" Multinomial Naive Bayes model Classification report:")
print("+----------------------------------------------------+")
print(classification_report(Y_test, MNB_Y_pred) )
print("Test accuarcy: {:.2f}%".format(MNB_acc*100))

#Get the confusion matrix.
print("\n")
print("+----------------------------------------------------+")
print(" Multinomial Naive Bayes model Confusion Matrix:")
print("+----------------------------------------------------+")
print(confusion_matrix(Y_test, MNB_Y_pred))

MNB_cm = confusion_matrix(Y_test, MNB_Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=MNB_cm)
disp.plot()

###############################################################################
                   # Support Vector Classifier model #
###############################################################################

from sklearn import svm

# classify using support vector classifier
svm = svm.SVC(kernel = 'linear', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(X_train, Y_train).predict_proba(X_test)

# perform classification and prediction on samples in x_test
Y_pred_svm = svm.predict(X_test)
SVM_acc = accuracy_score(Y_pred_svm, Y_test)


# Create classification report
SVM_cf = classification_report(Y_test, Y_pred_svm) 
print("\n")
print("+----------------------------------------------------+")
print(" Support Vector Classifier model Classification report:")
print("+----------------------------------------------------+")
print(classification_report(Y_test, Y_pred_svm) )
print("Test accuarcy: {:.2f}%".format(SVM_acc*100))

#Get the confusion matrix.
print("\n")
print("+----------------------------------------------------+")
print(" Support Vector Classifier model Confusion Matrix:")
print("+----------------------------------------------------+")
print(confusion_matrix(Y_test, Y_pred_svm))

SVM_cm = confusion_matrix(Y_test, Y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=SVM_cm)
disp.plot()

########## k-fold cross validation on Support Vector Classifier model #########

from sklearn.model_selection import cross_val_predict
kfold_predict = cross_val_predict(svm, X_train, Y_train, cv=10 )

SVM_kfold_acc = accuracy_score(Y_train,kfold_predict)

print("\n")
print("+----------------------------------------------------+")
print("k-fold cross validation with Support Vector Classifier model Classification report:")
print("+----------------------------------------------------+")
print(classification_report(Y_train, kfold_predict) )


print("Test accuarcy: {:.2f}%".format(SVM_kfold_acc*100))











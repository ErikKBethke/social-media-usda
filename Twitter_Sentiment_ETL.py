
# coding: utf-8

# # USDA Twitter Sentiment Analysis

# Performs sentiment analysis on Twitter data mentions of various USDA accounts
# 
# Author: Erik Bethke
# Last modified: 6.14.18

# ## Installation of nltk and data

# pip3 install nltk==3.2.4

# sudo python3 -m nltk.downloader all

# Importing modules and basic functions

# In[10]:


# Import modules for analysis
import nltk
import pandas as pd
import numpy as np
import glob
import sys
import datetime
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import TweetTokenizer

now = datetime.datetime.now()

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})


# ## Setting up training model

# ### Import positive tweet training data

# In[2]:


pos = []
with open("./pos_tweets.txt") as f:
    for i in f:
        pos.append([format_sentence(i), 'pos'])


# ### Import negative tweet training data

# In[3]:


#Establish negative tweet trainig dat
neg = []
with open("./neg_tweets.txt", encoding='utf-8') as f:
    for i in f:
        neg.append([format_sentence(i), 'neg'])


# Split labeled training data into the training and test data (80/20 split)

# In[4]:


training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
print("Building training data...")


# ### Build classifier

# In[5]:


classifier = NaiveBayesClassifier.train(training)
print("Building classifier...")


# Simple display of informative features from the training data (just for understanding)

# In[6]:


#classifier.show_most_informative_features()


# ### Test training data against a text string

# Simply for showing it works

# In[7]:


#example_pos = ("What a great date")
#example_neg = ("Don't trust anyone!")

#print('This example is positive and results in: ' + classifier.classify(format_sentence(example_pos)))
#print('This example is negative and results in: ' + classifier.classify(format_sentence(example_neg)))


# See the accuracy of our test data

# In[8]:


#print(accuracy(classifier, test))


# ## USDA Twitter Data

# ### Import USDA Twitter Data

# Opening in pandas dataframe

# In[9]:


#df = pd.read_excel('./Twitter_Final_18062018.xlsx', 'Sheet1')
print("Importing Twitter Data...")
csv = glob.glob('*Twitter_Full*')
csv = './' + csv[0]
print("Preparing " + csv + "...")
df = pd.read_csv(csv, sep='|', error_bad_lines=False, warn_bad_lines=True, encoding='latin-1')

#df = pd.read_csv('./Twitter_Full_18062018_parsed.csv')


# Run classifier on tweet texts, check for errors

# In[10]:


#classifier.classify(df['nltk'])
tweetSent = []
for tweet in df['tweetText']:
    try:
        tweetSent.append(classifier.classify(format_sentence(tweet)))
    except:
        tweetSent.append('error')       


# Push tweetSent to dataframe

# In[11]:


df['tweetSent'] = tweetSent
df = df.fillna('')


# ### Split tweet text, create new dataframe for each word in each tweet with associated sentiment

# Prepare new dataframe df_word for words

# In[12]:


# New dataframe for individual words
df_word_cols = list(df.columns)
df_word_cols.append('tweetWord')

df_word = pd.DataFrame(columns = df_word_cols)

# Split words into list format, set up dataframe locators
wordList = df['tweetText'].str.split(' ')


# Populate dataframe df_word

# In[13]:


df_word_vals = []
df_loc = 0
word_loc = 0

print("Generating parsed word file...")
# Parse through all word lists, then parse through all words to populate df_word
for words in wordList:
    for word in words:
        #print('word # ' + str(word_loc) + ' loc # ' + str(df_loc))
        #print(word)
        # Populate df_word with all basic values
        #df_word_fill = df.iloc[df_loc].append(word)
        #df_word = df_word.append(df.iloc[df_loc], ignore_index=True)
        df_append = df.iloc[df_loc]
        df_word = df_word.append(df_append, ignore_index=True)
        # this needs work to replace NaN with text
        try:
            df_word.at[word_loc, 'tweetWord'] = word
        except:
            df_word = df_word.replace(np.nan, '', regex=True)
            df_word.at[word_loc, 'tweetWord'] = word
        word_loc += 1
    df_loc += 1


# Combine all columns into 1 for improved filesize

# In[14]:


df_word['Compressed'] = df_word[df_word.columns[0:]].apply(lambda x: '==='.join(x.astype(str)),axis=1)
#df_word.dtypes


# In[15]:


#df_word.columns


# In[16]:


df_word_out = df_word[['tweetId', 'tweetDate', 'tweetWord', 'tweetSent', 'attitude']].copy()


# Checking error columns

# In[17]:


#df_error = df[df.tweetSent == 'error']
#df_error['tweetId']


# Outputting to CSV

# In[15]:


print("Writing output files...")
dmy =  str(now.month) + "_" + str(now.day) + "_" + str(now.year)
df.to_csv('./Twitter_PythonSentiment_' + dmy + '.csv', encoding='utf-8', index=False)
df_word_out.to_csv('./Twitter_PythonSentiment_Word_' + dmy + '.csv', encoding='utf-8', index=False)


# In[19]:


#df_backup.to_csv('./BackupTest.csv', encoding='utf-8')


# ### For Sampling

# In[20]:


#df_sample = df.sample(200)
#df_sample.to_csv('./Twitter_Sample_test.csv', encoding='utf-8')


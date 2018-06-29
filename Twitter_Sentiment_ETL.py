
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

# In[282]:


# Import modules for analysis
import nltk
import pandas as pd
import numpy as np
import glob
import sys
import string
import datetime
import shutil
from pathlib import Path
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import TweetTokenizer

now = datetime.datetime.now()
dmy =  str(now.month) + "_" + str(now.day) + "_" + str(now.year)

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})


# ## Setting up training model

# ### Import positive tweet training data

# In[5]:


pos = []
with open("./pos_tweets.txt") as f:
    for i in f:
        pos.append([format_sentence(i), 'pos'])


# ### Import negative tweet training data

# In[6]:


#Establish negative tweet trainig dat
neg = []
with open("./neg_tweets.txt", encoding='utf-8') as f:
    for i in f:
        neg.append([format_sentence(i), 'neg'])


# Split labeled training data into the training and test data (80/20 split)

# In[7]:


training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
print("Building training data...")


# ### Build classifier

# In[8]:


classifier = NaiveBayesClassifier.train(training)
print("Building classifier...")


# Simple display of informative features from the training data (just for understanding)

# In[9]:


#classifier.show_most_informative_features()


# ### Test training data against a text string

# Simply for showing it works

# In[10]:


#example_pos = ("What a great date")
#example_neg = ("Don't trust anyone!")

#print('This example is positive and results in: ' + classifier.classify(format_sentence(example_pos)))
#print('This example is negative and results in: ' + classifier.classify(format_sentence(example_neg)))


# See the accuracy of our test data

# In[11]:


#print(accuracy(classifier, test))


# ## USDA Twitter Data

# ### Import USDA Twitter Data

# Opening in pandas dataframe

# In[270]:


print("Importing Twitter Data...")
csv = glob.glob('*Twitter_Full*')
csv = './' + csv[0]
print("Preparing " + csv + "...")
df = pd.read_csv(csv, sep='|', error_bad_lines=False, warn_bad_lines=True, encoding='latin-1')

#df = pd.read_csv('./Twitter_Full_18062018_parsed.csv')
#df = pd.read_excel('./Twitter_Final_18062018.xlsx', 'Sheet1')


# Creating a data frame to hold the file name and date, to check against existing data in master list (and later for export)

# In[345]:


df_fileList = pd.DataFrame(columns = ['FileName', 'Date'])
df_fileList.loc[(len(df_fileList))] = [csv, dmy]


# Checking for Twitter Master, then checking if analysis has already been completed

# In[363]:


master = Path('./Twitter_Master.xlsx')
if master.is_file():
    print('Comparing file to list of completed sentiment analyses for duplicates...')
    # Read master file list, append df_fileList to master list
    df_master_fileList = pd.read_excel(master, 'Sheet2', index=False)
    if (df_master_fileList['FileName'].str.contains(csv).sum() > 0 and df_master_fileList['Date'].str.contains(dmy).sum()):
        print('Sentiment analysis already completed on this file. Exiting...')
        print('Please remove duplicate file from root directory.')
        sys.exit()
    else:
        print('No duplicate detected. Continuing...')
else:
    print('No duplicate detected. Continuing...')


# Run classifier on tweet texts, check for errors

# In[255]:


#classifier.classify(df['nltk'])
tweetSentiment = []
for tweet in df['tweetText']:
    try:
        tweetSentiment.append(classifier.classify(format_sentence(tweet)))
    except:
        tweetSentiment.append('error')       


# Push tweetSentiment to dataframe

# In[256]:


df['tweetSentiment'] = tweetSentiment
df = df.fillna('')


# ### Import stopwords data and push to data frame

# In[257]:


print('Importing stopwords...')
df_stop = pd.read_excel('./stopwords.xlsx', header=None, names=['stop'])


# Push to list

# In[258]:


stopList = df_stop['stop'].values.tolist()


# ### Split tweet text, create new dataframe for each word in each tweet with associated sentiment

# Prepare new dataframe df_word for words

# In[259]:


# New dataframe for individual words
df_word_cols = list(df.columns)
df_word_cols.append('tweetWord')

df_word = pd.DataFrame(columns = df_word_cols)

# Split words into list format, set up dataframe locators
wordList = df['tweetText'].str.split(' ')


# Populate dataframe df_word

# In[260]:


df_word_vals = []
df_loc = 0
word_loc = 0
lowerExcept = ['US', 'U.S.']

# Build translator using string module to remove punctuation from words (excluding @ or #)
translatorWords = string.punctuation
translatorBuild = str.maketrans('', '', '@#')
translatorWords = translatorWords.translate(translatorBuild)
translator = str.maketrans('', '', translatorWords)

print("Generating parsed word file...")

# Parse through all word lists, then parse through all words to populate df_word
for words in wordList:
    for word in words:
        # Drop punctuation from word, ignoring # or @
        # Then convert to lower case if not in list of exceptions
        if word in lowerExcept:
            wordDrop = word.translate(translator)
        else:
            wordDrop = word.translate(translator).lower()
        
        # Check word against stop list
        if wordDrop in stopList:
            pass
        #something broken here
        elif wordDrop not in stopList:
            # something going wrong in this section
            df_append = df.iloc[df_loc]
            df_word = df_word.append(df_append, ignore_index=True)
            try:
                df_word.at[word_loc, 'tweetWord'] = wordDrop
            except:
                df_word = df_word.replace(np.nan, '', regex=True)
                df_word.at[word_loc, 'tweetWord'] = wordDrop
            word_loc += 1
    df_loc += 1


# Combine all columns into 1 for improved filesize

# In[261]:


df_word_out = df_word[['tweetId', 'tweetDate', 'tweetWord', 'tweetSentiment']].copy()


# Outputting to CSV, archives Full_Data, Sentence Sentiment, and Word Sentiment appropriately

# In[279]:


print("Writing output files...")
df.to_csv('./archive/Sentence_Sentiment/Twitter_PythonSentiment_' + dmy + '.csv', encoding='utf-8', index=False)
df_word_out.to_csv('./archive/Word_Sentiment/Twitter_PythonSentiment_Word_' + dmy + '.csv', encoding='utf-8', index=False)
shutil.move(csv, './archive/Full_Data/')


# ### Exporting to Master Data

# Check for Twitter_Master Excel, then write to appropriate locations

# In[344]:


master = Path('./Twitter_Master.xlsx')
if master.is_file():
    # Read master data, append df_word_out to master list
    df_master = pd.read_excel(master, 'Sheet1', index=False)
    df_master = df_master.append(df_word_out, ignore_index=True)
    
    # Read master file list, append df_fileList to master list
    df_master_fileList = pd.read_excel(master, 'Sheet2', index=False)
    df_master_fileList = df_master_fileList.append(df_fileList)
    
    writer = pd.ExcelWriter('./Twitter_Master.xlsx')
    df_master.to_excel(writer,'Sheet1', index=False)
    df_master_fileList.to_excel(writer,'Sheet2', index=False)
    writer.save()
else:
    writer = pd.ExcelWriter('./Twitter_Master.xlsx')
    df_word_out.to_excel(writer,'Sheet1', index=False)
    df_fileList.to_excel(writer,'Sheet2', index=False)
    writer.save()
    
print('Job completed')


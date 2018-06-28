# social-media-usda

## Setting up the environment:
1. Install Python 3 on your machine ([Anaconda distribution](https://www.anaconda.com/) works well)
2. In your project root directory, clone the repo

`git clone
https://github.com/ErikKBethke/social-media-usda`

3. (Optional) If not already installed, install pip3

`sudo apt-get install python3-pip`

4. Install Natural Language Toolkit ([nltk](https://www.nltk.org/)) and additional functionality

`pip3 install nltk==3.2.4`

`sudo python3 -m nltk.downloader all`



## File setup
1. Training data text files [neg_tweets.txt, pos_tweets.txt] must be in the root folder
2. The python file Twitter_Sentiment_ETL.py must be in the root folder
3. The USDA Twitter data feed must come in formatting established by PJ, and must have "Twitter_Full" in the file name. This file must be in the root folder

## Usage
1. Positive and negative tweet training data is fed into a Naive Bayes Classifier
2. USDA social media data is pulled into a pandas data frame
3. The Naive Bayes classification runs sentiment analysis on each Tweet, and sentiment is appended to the data frame
4. Each sentence is parsed through, creating a new data frame that contains rows for each word of every Tweet with associated data (sentiment, date, etc.)
5. Two files are output:
* Twitter_PythonSentiment_DATE.csv contains rows for each sentence
* Twitter_PythonSentiment_Word_DATE.csv contains rows for each word

### Notes/To-Do
1. Improve training data to be more catered to USDA tweet Language
2. Improve automation
3. Better read/write functionality for data

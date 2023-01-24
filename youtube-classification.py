import pandas as pd
import nltk
import re
import random
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
 
import warnings
warnings.simplefilter(action='ignore')

# NLTK extra files
nltk.download('stopwords')
nltk.download('punkt')

# load dataset
yt = pd.read_csv('youtube-videos.csv')

print("|> 1/3 - Dataset loaded")

# delete useless column
del yt['file']
del yt['Unnamed: 0']
del yt['Unnamed: 0.1']

# delete incomplete records
yt.dropna(axis=0, how='any',inplace=True)

# merge columns and delete old ones
yt['Tokens'] = yt['Title'] + ' ' + yt['Description']
del yt['Title']
del yt['Description']

# text clean
def remove_html(text):
    bs = BeautifulSoup(text, "html.parser")
    return bs.get_text()

def remove_urls(text):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def remove_stopwords(text, is_lower_case=False):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokenizer = ToktokTokenizer()

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

yt['Tokens'] = yt['Tokens'].apply(remove_html)
yt['Tokens'] = yt['Tokens'].apply(remove_urls)
yt['Tokens'] = yt['Tokens'].apply(remove_special_characters)
yt['Tokens'] = yt['Tokens'].apply(remove_stopwords)

# stemming
def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

yt['Tokens'] = yt['Tokens'].apply(stemmer)

# tokenization
yt['Tokens'] = yt['Tokens'].apply(nltk.word_tokenize)

# data preparation for training
categories = ['travel', 'science and technology', 'food', 'manufacturing', 'history', 'art and music', 'nature', 'sports', 'adventure']

old_x = [i for i in yt['Tokens'].to_list()]

x = ['' for _ in old_x]

w = 0
for i in old_x:
    for j in i:
        x[w] += j + ' '
    w+=1
    
y = [None for i in range(28509)]

j = 0
for i in yt['Category'].to_list():
    if i == 'travel':
        y[j] = 0
    if i == 'science and technology':
        y[j] = 1
    if i == 'food':
        y[j] = 2
    if i == 'manufacturing':
        y[j] = 3
    if i == 'history':
        y[j] = 4
    if i == 'art and music':
        y[j] = 5
    if i == 'nature':
        y[j] = 6
    if i == 'sports':
        y[j] = 7
    if i == 'adventure':
        y[j] = 8
    j+=1

print("|> 2/3 - Dataset cleaned")

#records shuffle
temp = list(zip(x, y))

random.shuffle(temp)

x, y = zip(*temp)
x, y = list(x), list(y)

#train set separation
train_s = x

train_t = y

# pipeline generation
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),])

# cross-validation
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# grid-search
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, cv=cv, n_jobs=-1)

# training
gs_clf = gs_clf.fit(train_s, train_t)

print("|> 3/3 - Training complete")

# single-video test
while True:
    print("|> Insert the title and description of the video as a single string:")
    test = input("|>> ")

    test = pd.Series(test)

    test.apply(remove_html).apply(remove_urls).apply(remove_special_characters).apply(remove_stopwords).apply(stemmer).apply(nltk.word_tokenize)

    res = gs_clf.predict(test)

    if res[0] == 0:
        res = 'travel'
    if res[0] == 1:
        res = 'science and technology'
    if res[0] == 2:
        res = 'food'
    if res[0] == 3:
        res = 'manufacturing'
    if res[0] == 4:
        res = 'history'
    if res[0] == 5:
        res = 'art and music'
    if res[0] == 6:
        res = 'nature'
    if res[0] == 7:
        res = 'sports'
    if res[0] == 8:
        res = 'adventure'

    print(res)
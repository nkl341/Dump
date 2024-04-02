# program		Team1_Movie_Sentiment.py
# purpose	    Develop an algorithm that is able to accurately predict whether a movie
#               review is positive or negative using machine learning
# usage         algorithm
# notes         1) used data set from kaggle "IMDB Dataset of 50k Movie Review"
#               2) needed to download nltk.corpus and ntlk.stem onto computer
#               3) condensed the dataset down to 1000 samples as it still produced
#                  82% accuracy and did not take as long to load
# date			11/22/2022-11/29/2022
# programmer    C. Novelo and Alex Luebbert (with help using Kaggle examples)

#------------------------------------------------------------------------------------------------#

#importing required libraries
import pandas as pd
import numpy as np
import re
#LabelEncoder converts the categorical data into quantitative data
from sklearn.preprocessing import LabelEncoder
import nltk
#nltk.download() #download Porterstemmer if needed
from nltk.stem import PorterStemmer
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


#nltk.download('words')
words = set(nltk.corpus.words.words())

#These "stop words" are words that are clearly neutral, so they will be ignored.
stopWords = ['I', 'me', 'my' 'myself', 'we', 'our', 'ours', 'ourselves', 'not','you',"you're", "you've","you'll","you'd","your","yours","yourself","yourselves","he","him","his","himself","she","she's","her","hers","herself","it","it's","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","that'll","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","can","will","just","don't","should","should've","now","aren't","couldn't","didn't","doesn't","hadn't","hasn't","haven't","isn't","needn't","shan't","shouldn't","wasn't","weren't","won't","wouldn't", "movie", "thought", "one", "two", "three", "spoilers", "spoiler", "but", "think", "film", "plot", "premise", "setting"]

#loading to show top 5 rows of reviews
df = pd.read_csv("imbd1000Data.csv")
print(df.head())

#either 'positive' or 'negative'
df['sentiment'].unique()

#of the 1000 sample 501 are positive, 499 are negative reviews
print(df['sentiment'].value_counts())

#applying the label encoding to make the negative=0 and positive=1
label = LabelEncoder()
df['sentiment'] = label.fit_transform(df['sentiment'])
(df.head)

#dividing into independent and dependent
X = df['review'] #independent
y = df['sentiment'] #dependent


ps = PorterStemmer()
corpus = []
for i in range(len(X)):
    (i)
    review = re.sub("[^a-zA-Z]"," ", X[i])
    review = review.lower()
    review = review.split()
    
#removing all special characters and stopwords (common words with no sentiment) 
#and "stemming" (converting words to root words)

    review = [ps.stem(word) for word in review if word not in set(stopWords)]
    review = " ".join(review)
    review = " ".join(w for w in nltk.wordpunct_tokenize(review) if w.lower() in words or not w.isalpha())
    corpus.append(review)

#apply TfidVectorizer to convert text data into vectors
cv = TfidfVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

#split data into train and test sets
X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size=0.3 , random_state=101)
print((X_train.shape , X_test.shape , Y_train.shape , Y_test.shape))

#define naive-bayes model
mnb = MultinomialNB()
mnb.fit(X_train , Y_train)

#test model using the test data
pred = mnb.predict(X_test)

#checking accurary score, the confusion matrix, and classification report
print(accuracy_score(Y_test , pred))
print(confusion_matrix(Y_test , pred))
print(classification_report(Y_test , pred))

#checking difference between actual and predicted data (in column)
print(pd.DataFrame(np.c_[Y_test , pred] , columns=["Actual" , "Predicted"]))

#defining new function to test model
def test_model(sentence):
    sen = cv.transform([sentence]).toarray()
    if mnb.predict(sen)[0] == 1:
        return 'Positive review'
    else:
        return 'Negative review'
    
#testing the model
sen='This is the best movie i have ever seen, i want to watch it over and over again'
res=test_model(sen)
print(res)

sen1='This movie is chill i like it but at times it is a bit slow but it is still good overall'
res1=test_model(sen1)
print(res1)

sen2='I was really hopeful for this movie but after watching it i really just did not like it'
res2=test_model(sen2)
print(res2)

sen3='I hate this movie'
res3=test_model(sen3)
print(res3)

sen4="This is my new favorite movie"
res4=test_model(sen4)
print(res4)
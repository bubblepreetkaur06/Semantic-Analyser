# Semantic-Analyser
Data Cleaning :
`````
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

all_stopwords=stopwords.words('english') 
all_stopwords.remove('not')
`````
`````
corpus=[]

for i in range(0,900):    # row by row to clean
  review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) # sub is for pattern subsitutuion, here we are selecting not upper case and lower case letters. if others characters found replac them with space.
  review=review.lower()
  review=review.split() # split the review into words/tokens.
  review=[ps.stem(word) for word in review if not word in set(all_stopwords)] # stemmer reduce words to their base forlike running to run
  review=' '.join(review)
  corpus.append(review) ## appending each cleaned review to corpus
`````
Data Transformation: 

``````
# CountVectorizer helps to represent text data numerically.
from sklearn.feature_extraction.text import CountVectorizer # converts a collection of text documents into matrix of fatures.
cv= CountVectorizer(max_features=1420) # taking top 1420 tokens and dropping rest
``````

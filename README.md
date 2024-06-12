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

for i in range(0,900):    
  review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) 
  review=review.lower()
  review=review.split() 
  review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
  review=' '.join(review)
  corpus.append(review) 
`````
Data Transformation: 

``````
from sklearn.feature_extraction.text import CountVectorizer 
cv= CountVectorizer(max_features=1420)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,-1].values
``````
``````
# saving bag of words dictionary
import pickle 
bow_path='/content/sample_data/bag_of_words.pkl'
pickle.dump(cv,open(bow_path,"wb")) 
``````
Model Selection and fitting :

``````
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
``````
``````
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)
``````
``````
# Exporting NB classifier for later use
import joblib   
joblib.dump(classifier,'/content/sample_data/Classifier_store') 
``````
Model Performance :

````
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,y_pred)
print(cm)

accuracy_score(Y_test,y_pred)
````
Prediction Model:
`````
import numpy as np
import pandas as pd

dataset2=pd.read_csv('/content/sample_data/Fresh_restaurant.tsv', delimiter='\t',quoting=3)
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps1=PorterStemmer()

all_stopwords1=stopwords.words('english')
all_stopwords1.remove('not')
`````
`````
corpus2=[]

for i in range(0,100):
  review_fresh=re.sub('[^a-zA-Z]',' ',dataset2['Review'][i])
  review_fresh=review_fresh.lower()
  review_fresh=review_fresh.split()
  review_fresh=[ps1.stem(word1) for word1 in review_fresh if not word1 in set(all_stopwords1)]
  review_fresh=' '.join(review_fresh)
  corpus2.append(review_fresh)
`````
`````
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cv_file='/content/sample_data/bag_of_words.pkl'
cv1=pickle.load(open(cv_file,"rb"))
X_fresh=cv1.transform(corpus2).toarray()
X_fresh.shape
`````
`````
y_pred_fresh=classifier1.predict(X_fresh)
print(y_pred_fresh)
dataset2['Predicted_Label']= y_pred_fresh.tolist()
dataset2.head(5)
`````
![image](https://github.com/bubblepreetkaur06/Semantic-Analyser/assets/164672202/06a31575-22f3-4d2e-be2f-20718b16aa82)



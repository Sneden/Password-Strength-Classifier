#Logistic Regression with metrics


# Import the necessary Libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid') 
# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For creating a pipeline
from sklearn.pipeline import Pipeline

# Classifier Model (Logistic Regression)
from sklearn.linear_model import LogisticRegression


# Read the File
data = pd.read_csv('training.csv')

# Features which are passwords
features = data.values[:, 1].astype('str')

# Labels which are strength of password
labels = data.values[:, -1].astype('int')

# Apply a list of transforms and a final estimator
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('logisticRegression',LogisticRegression(multi_class='multinomial', solver='sag')),
])
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.2, random_state=1)
# Fit the Model
classifier_model.fit(features_train, labels_train)

preds = classifier_model.predict(features_test)

# Training Accuracy
print('Training Accuracy: ',classifier_model.score(features_train, labels_train))

#Test accuracy
accuracy = metrics.accuracy_score(labels_test, preds)
print("Accuracy : ", accuracy, "\n")
#Test precision
precision_positive = metrics.precision_score(labels_test, preds, average='weighted')
precision_negative = metrics.precision_score(labels_test, preds, average='weighted')
print("Precision : ", precision_positive," , ", precision_negative, "\n")
#Test f1 score
f1_positive = metrics.f1_score(labels_test, preds, average='weighted')
f1_negative = metrics.f1_score(labels_test, preds, average='weighted')
print("F1 score : ", f1_positive," , ", f1_negative, "\n")
#Test jaccard
jaccard = metrics.jaccard_score(labels_test, preds, average='micro')
print("Jaccard coeficient : ", jaccard, "\n")
#Test hamming loss
hamming_loss = metrics.hamming_loss(labels_test, preds)
print("Hamming loss : ", hamming_loss,"\n")



# Save model for Logistic Regression

pickle.dump(classifier_model, open('LR_model.sav', 'wb'))

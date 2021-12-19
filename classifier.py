# main code 


import pickle
import numpy as np

    
Password1 = input("Enter your password: ")
Password=[Password1]

# load models
DT_Model = pickle.load(open('DT_model.sav', 'rb'))

LR_Model = pickle.load(open('LR_model.sav', 'rb'))

RF_Model = pickle.load(open('RF_model.sav', 'rb'))

# Predict the strength
DT_Test = DT_Model.predict(Password)
DT_Test_prob = DT_Model.predict_proba(Password)

LR_Test = LR_Model.predict(Password)
LR_Test_prob = LR_Model.predict_proba(Password)

RF_Test = RF_Model.predict(Password)
RF_Test_prob = RF_Model.predict_proba(Password)

# print resuls with prediction probability
if DT_Test == 0:
  print("Decision Tree : weak password")
  print (np.array(DT_Test_prob*100, int), '%','\n')
elif DT_Test == 1:
  print("Decision Tree : medium password")
  print (np.array(DT_Test_prob*100, int), '%','\n')
else:
  print("Decision Tree : strong password")
  print (np.array(DT_Test_prob*100, int), '%','\n')



if LR_Test == 0:
  print("Logistic Regression : weak password")
  print (np.array(LR_Test_prob*100, int), '%','\n')
elif LR_Test == 1:
  print("Logistic Regression : medium password")
  print (np.array(LR_Test_prob*100, int), '%','\n')
else:
  print("Logistic Regression : strong password")
  print (np.array(LR_Test_prob*100, int), '%','\n')


if RF_Test == 0:
  print("Random Forest : weak password")
  print (np.array(RF_Test_prob*100, int), '%','\n')
elif RF_Test == 1:
  print("Random Forest : medium password")
  print (np.array(RF_Test_prob*100, int), '%','\n')
else:
  print("Random Forest : strong password")
  print (np.array(RF_Test_prob*100, int), '%','\n')

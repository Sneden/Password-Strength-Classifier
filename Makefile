all: DT_model LR_model RF_model main

DT_model: decisiontree.py 
	python3 decisiontree.py

LR_model: logisticregression.py
	python3 logisticregression.py

RF_model: randomforest.py
	python3 randomforest.py
	
main: classifier.py
	python3 classifier.py

clean:
	rm *.sav

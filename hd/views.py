from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def home(request):
    return render(request,'index.html',{"predicted":""})



def predict(request):
    exp1 = float(request.GET['exp1'])
    exp2 = float(request.GET['exp2'])
    exp3 = float(request.GET['exp3'])
    exp4 = float(request.GET['exp4'])
    exp5 = float(request.GET['exp5'])
    exp6 = float(request.GET['exp6'])
    exp7 = float(request.GET['exp7'])
    exp8 = float(request.GET['exp8'])
    exp9 = float(request.GET['exp9'])
    exp10 = float(request.GET['exp10'])
    exp11 = float(request.GET['exp11'])
    exp12 = float(request.GET['exp12'])
    exp13 = float(request.GET['exp13'])
    rawdata = staticfiles_storage.path('heart.csv')
    dataset = pd.read_csv(rawdata)
    X = dataset[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
    y = dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    yet_to_predict = np.array([[exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8,exp9,exp10,exp11,exp12,exp13]])
    y_pred = model.predict(yet_to_predict)
    accuracy = model.score(X_test, y_test)
    accuracy = accuracy*100
    accuracy = int(accuracy)
    return render(request,'index.html',{"predicted":y_pred[0],"exp1":exp1,"exp2":exp2,"exp3":exp3,"exp4":exp4,"exp5":exp5,"exp6":exp6,"exp7":exp7,"exp8":exp8,"exp9":exp9,"exp10":exp10,"exp11":exp11,"exp12":exp12,"exp13":exp13})
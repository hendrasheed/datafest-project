# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:48:05 2019

@author: HendR
"""

#PLAYER TREE MODELS + FEATURE IMPORTANCE CALCULATIONS

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import display
from mlxtend.preprocessing import standardize
import numpy as np

wellnessDF = pd.read_csv('wellnessCopy.csv')

#Converting dates to ints for the wellness dataframe
count = 0
for index, row in wellnessDF.iterrows():
    stringDate = row['Date']
    dateArray = stringDate.split("/")
    if (len(dateArray[0]) == 1):
        monthStr = "0" + dateArray[0]
    else:
        monthStr = dateArray[0]
    if (len(dateArray[1]) == 1):
        dayStr = "0" + dateArray[1]
    else:
        dayStr = dateArray[1]
    yearStr = dateArray[2]
        
    convStrDate = yearStr + monthStr + dayStr
        
    intDate = int(convStrDate)
        
    wellnessDF.iat[count, wellnessDF.columns.get_loc('Date')] = intDate
        
    count += 1

#Making a dataframe for each player
player1DF = wellnessDF[wellnessDF['PlayerID'] == 1]
player2DF = wellnessDF[wellnessDF['PlayerID'] == 2]
player3DF = wellnessDF[wellnessDF['PlayerID'] == 3]
player4DF = wellnessDF[wellnessDF['PlayerID'] == 4]
player5DF = wellnessDF[wellnessDF['PlayerID'] == 5]
player6DF = wellnessDF[wellnessDF['PlayerID'] == 6]
player7DF = wellnessDF[wellnessDF['PlayerID'] == 7]
player8DF = wellnessDF[wellnessDF['PlayerID'] == 8]
player9DF = wellnessDF[wellnessDF['PlayerID'] == 9]
player10DF = wellnessDF[wellnessDF['PlayerID'] == 10]
player11DF = wellnessDF[wellnessDF['PlayerID'] == 11]
player12DF = wellnessDF[wellnessDF['PlayerID'] == 12]
player13DF = wellnessDF[wellnessDF['PlayerID'] == 13]
player14DF = wellnessDF[wellnessDF['PlayerID'] == 14]
player15DF = wellnessDF[wellnessDF['PlayerID'] == 15]
player16DF = wellnessDF[wellnessDF['PlayerID'] == 16]
player17DF = wellnessDF[wellnessDF['PlayerID'] == 17]

players = [player1DF, player2DF, player3DF, player4DF, player5DF,
           player6DF, player7DF, player8DF, player9DF, player10DF,
           player11DF, player12DF, player13DF, player14DF, player15DF,
           player16DF, player17DF]

features = ["Soreness", "Desire", "Irritability", "SleepHours", "SleepQuality",
            "PainNum", "IllnessNum", "MenstruationNum"]

lists_of_lists = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
lists_of_lists[0] = [0,0,0,0,0,0,0,0]

sumAccuracy = 0
count = 1
for player in players:
    print("Player: " + str(count))
    X = player[features]
    y = player.FatigueBin
    
    #Splitting the data and modeling and testing from there
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, test_size=0.1)
        
    #A function to find the best accuracy with values of leaf nodes from 2-50
    def findBestAccuracy(train_X, val_X, train_y, val_y):
        model = DecisionTreeClassifier(max_leaf_nodes = 2, random_state=1)
        model.fit(train_X, train_y)
        valPredictions = model.predict(val_X)
        maxAccuracy = accuracy_score(val_y, valPredictions)
        maxX = 2
        for x in range(2, 50):
            model = DecisionTreeClassifier(max_leaf_nodes = x, random_state=1)
            model.fit(train_X, train_y)
            valPredictions = model.predict(val_X)
            if accuracy_score(val_y, valPredictions) > maxAccuracy:
                maxAccuracy = accuracy_score(val_y, valPredictions)
                maxX = x
        print("Best Num leaf nodes:")
        print(maxX)
# =============================================================================
#         print("Best accuracy:")
#         print(maxAccuracy)
# =============================================================================
        return maxX
        
    leaf = findBestAccuracy(train_X, val_X, train_y, val_y)
    
    model = DecisionTreeClassifier(max_leaf_nodes=leaf, random_state=1)
    model.fit(train_X, train_y)
    valPredictions = model.predict(val_X)
    MAE = mean_absolute_error(val_y, valPredictions)
    R2 = r2_score(val_y, valPredictions)
    print("MAE:")
    print(MAE)
    print()
    print("R2:")
    print(R2)
    print()
    print("Accuracy:")
    print(accuracy_score(val_y, valPredictions))
    print()
    sumAccuracy += accuracy_score(val_y, valPredictions)
    

    featureImps = model.feature_importances_
    print(dict(zip(features, featureImps)))
    print()
    
    lists_of_lists[count] = featureImps
    
    #######################CODE FOR DRAWING THE TREES#######################
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,
                    feature_names=features,
                    class_names=["1","2","3"],
                    filled=True, rounded=True,
                    special_characters=True,
                    impurity=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    i = Image(graph.create_png())
    display(i)
    #######################################################################
    
    count += 1
    
sumList = [sum(x) for x in zip(*lists_of_lists)]

for x in range(len(sumList)):
    sumList[x] = sumList[x] / 17

print()
AvgAccuracy = sumAccuracy / 17
print("Average Accuracy: " + str(AvgAccuracy))

print()
print(dict(zip(features, sumList)))
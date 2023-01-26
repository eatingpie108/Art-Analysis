#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:17:13 2022

@author: michaeldelarosa
"""


import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn import metrics   
from sklearn.metrics import silhouette_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso 
from sklearn.preprocessing import scale 
import seaborn as sn
from sklearn.model_selection import KFold
from scipy.stats import mannwhitneyu
from scipy.stats import kstest
import statistics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random

random.seed(12844336)
theArt = pd.DataFrame(pd.read_csv("theArt.csv"))
theData = pd.DataFrame(pd.read_csv("theData.csv", header = None))


#%% This is the code for answering question 1

# Since we want to just determine if the calssical vs. modern art are rated more than the other , we can aggregate those scores into one list

# Classic more than modern
modernArtScores = []
classicArtScores = []

# finding the indexes of the modern and classical art 

# Mann Whitney U test 

artType = theArt['Source (1 = classical, 2 = modern, 3 = nonhuman)'].tolist()
modernIndex = []
classicIndex =[]

for i in range(91):
    if artType[i] == 1:
        classicIndex.append(i)
    elif artType[i] == 2:
        modernIndex.append(i)

for i in modernIndex:
    ratingList = theData.iloc[:,i]
    for j in ratingList:
        modernArtScores.append(j)

for i in classicIndex:
    ratingList = theData.iloc[:,i]
    for j in ratingList:
        classicArtScores.append(j)
        
#Cleaning 
modernArtScores = np.array(modernArtScores)
#modernArtScores = modernArtScores[np.isfinite(modernArtScores)]

classicArtScores = np.array(classicArtScores)
#classicArtScores = classicArtScores[np.isfinite(classicArtScores)]

# Conducting the mann whitney u 
testStatistic, pvalue1 = mannwhitneyu(modernArtScores, classicArtScores, alternative = 'less')

print(pvalue1)

#%% This is the code for answering question 2

# KS test, diff

nonhumanIndex = []
nonhumanScores = []

for i in range(91):
    if artType[i] == 3:
        nonhumanIndex.append(i)

for i in nonhumanIndex:
    raingList = theData.iloc[:,i]
    for j in ratingList:
        nonhumanScores.append(j)
#Cleaning 
nonhumanScores = np.array(nonhumanScores)
#nonhumanScores = nonhumanScores[np.isfinite(nonhumanScores)]

statistic, pvalue = kstest(nonhumanScores,modernArtScores)
print(pvalue)



#%% This is the code for answering question 3

# Mann U

maleRatings = []

femaleRatings = []

for i in range(300):
    if theData.iat[i,216] == 1:
        theirData = theData.iloc[i].tolist()
        for j in range(91):
            maleRatings.append(theirData[j])
    elif theData.iat[i,216] == 2:
        theirData = theData.iloc[i].tolist()
        for j in range(91):
            femaleRatings.append(theirData[j])

statsitic, pvalue = mannwhitneyu(maleRatings,femaleRatings, alternative = 'less')
print(pvalue)


#%% This is the code for answering question 4

#need to find max years of art background 

# KS test, difference 

artEduCol = theData.iloc[:,218]

 #print(max(artEduCol)) # We got 3 lol

#Now we parse to gather data! 

zeroExp = []
oneExp = []
twoExp = []
threeExp = []

for i in range(300):
    if theData.iat[i,218] == 0:
        theirData = theData.iloc[i].tolist()
        for j in range(91):
            zeroExp.append(theirData[j])
    elif theData.iat[i,218] == 1:
        theirData = theData.iloc[i].tolist()
        for j in range(91):
            oneExp.append(theirData[j])
    elif theData.iat[i,218] == 2:
        theirData = theData.iloc[i].tolist()
        for j in range(91):
            twoExp.append(theirData[j])
    elif theData.iat[i,218] == 3:
        theirData = theData.iloc[i].tolist()
        for j in range(91):
            threeExp.append(theirData[j])
# Cleaning...
zeroExp = np.array(zeroExp)
#zeroExp = zeroExp[np.isfinite(zeroExp)]

oneExp = np.array(oneExp)
#oneExp = oneExp[np.isfinite(oneExp)]

twoExp = np.array(twoExp)
#twoExp = twoExp[np.isfinite(twoExp)]

threeExp = np.array(threeExp)
#threeExp = threeExp[np.isfinite(threeExp)]

someExp = np.concatenate((oneExp, twoExp, threeExp))

statistic, pvalue = kstest(zeroExp, someExp)
print(pvalue)


#%% This is the code for answering question 5

energyRatings = theData.iloc[:,91:182]

        
meanEnergyList = []
medianEnergyList = []

for i in range(300):
    tempList = energyRatings.iloc[i,:]
    meanEnergyList.append(tempList.mean())
    medianEnergyList.append(statistics.median(tempList))
        

ratings = theData.iloc[:,:91]
meanRatingsList = []
medianRatingsList = []
for i in range(300):
    tempList = ratings.iloc[i,:]
    meanRatingsList.append(tempList.mean())
    medianRatingsList.append(statistics.median(tempList))

flatNRG = energyRatings.to_numpy().flatten()
flatrating = ratings.to_numpy().flatten()


# Cross Validation with K-fold where K = 10

kfold1 = KFold(10)
totalRMSE = []
for train, test in kfold1.split(energyRatings):
    trainingX = []
    trainingY = []
    testingX = []
    testingY = []
    for ii in train:
        trainingX.append(meanEnergyList[ii])
        trainingY.append(medianRatingsList[ii])
    for jj in test:
        testingX.append(meanEnergyList[jj])
        testingY.append(medianRatingsList[jj])
    lasso = Lasso(max_iter = 10000, normalize = True)
    lasso.set_params(alpha = 0.1)
    lasso.fit(scale(pd.DataFrame(trainingX)),trainingY)
    predictionLass = lasso.predict(pd.DataFrame(testingX))
    totalRMSE.append(mean_squared_error(testingY,predictionLass,squared = False))
print('mean RMSE for models',statistics.mean(totalRMSE))


# Plotting the data 
plt.scatter(meanEnergyList,meanRatingsList)
plt.title("Mean Energies vs. Mean Painting Ratings")
plt.xlabel("Mean Energies")
plt.ylabel("Mean Painting Ratings")


#%% This is the code for snwering question 6

# now we make a column of predictors from the demographic informations, which are age, gender, political orientation and art Education
# Since each of these has an individual value per person, we cna just concat the columns onto a new predictors matrix

predictorMatrix = pd.DataFrame(meanEnergyList)

addThis = theData.iloc[:,215:220]

predictorMatrix = pd.concat([predictorMatrix,addThis],axis =1)

#Cleaning (Removing the people with missing data, since that won't help us and its not THAT much data)
fixIT = pd.concat([predictorMatrix,pd.DataFrame(meanRatingsList)],axis =1)
fixIT = pd.concat([fixIT,pd.DataFrame(medianRatingsList)],axis = 1)
fixIT = fixIT.dropna()

NEWmeanRatingsList = fixIT.iloc[:,6].copy()
NEWmedianRatingsList = fixIT.iloc[:,7]
predictorMatrix = fixIT.iloc[:,:5]

fullTrainingT, fullTestingT, ratingsTraining2T,ratingsTesting2T = train_test_split(predictorMatrix, NEWmedianRatingsList, test_size = .20, random_state=12844336)

numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(pd.DataFrame(fullTrainingT),ratingsTraining2T) #bagging numTrees trees

# Use model to make predictions:
predictions = clf.predict(pd.DataFrame(fullTestingT))

# Assess model accuracy:
modelAccuracy = accuracy_score(ratingsTesting2T,predictions)
print('Random forest model accuracy:',modelAccuracy)

print(addThis.corr()) # Since all of their correlations are relatively low, we can assume independence for all and use all vars 
heatmap = addThis.corr().to_numpy()
sn.heatmap(heatmap, xticklabels=['Age','Gender', 'Political Orientation', 'Art Education'], yticklabels = ['Age','Gender', 'Political Orientation', 'Art Education'])
plt.title("Confusion Matrix of Predictors")

#Doing K-fold for this one 
kfold2 = KFold(10)
rmseList = []

for train, test in kfold1.split(predictorMatrix):
    trainingX = pd.DataFrame()
    trainingY = []
    testingX = pd.DataFrame()
    testingY = []
    for ii in train:
        print(ii)
        trainingX = pd.concat([trainingX,pd.DataFrame(predictorMatrix.iloc[ii]).transpose()])
        trainingY.append(NEWmedianRatingsList.iloc[ii])
    for jj in test:
        testingX = pd.concat([testingX,pd.DataFrame(predictorMatrix.iloc[jj]).transpose()])
        testingY.append(NEWmedianRatingsList.iloc[jj])
    lasso = Lasso(max_iter = 10000, normalize = True)
    lasso.set_params(alpha = 0.1)
    lasso.fit(scale(pd.DataFrame(trainingX)),trainingY)
    predictionLass = lasso.predict(pd.DataFrame(testingX))
    rmseList.append(mean_squared_error(testingY,predictionLass,squared = False))
print('mean RMSE for models',statistics.mean(rmseList))


#%% This is the code for answering question 7

energyRatings = theData.iloc[:,91:182]

        
q7meanEnergyList = []

for i in range(91):
    tempList = energyRatings.iloc[:,i]
    q7meanEnergyList.append(tempList.mean())
        

ratings = theData.iloc[:,:91]

q7meanRatingsList = []
for i in range(91):
    tempList = ratings.iloc[:,i]
    q7meanRatingsList.append(tempList.mean())
    
    
#Make a basic plot! 
plt.scatter(q7meanEnergyList,q7meanRatingsList)
plt.title("Mean Energy per painting against Mean Rating per painting")

numClusters = 9 
sSum = np.empty([numClusters,1])*np.NaN 

combinedData = pd.concat([pd.DataFrame(q7meanEnergyList),pd.DataFrame(q7meanRatingsList)],axis =1)



for ii in range(2, numClusters+2): 
    kMeans = KMeans(n_clusters = int(ii)).fit(combinedData) 
    cId = kMeans.labels_ 
    cCoords = kMeans.cluster_centers_ 
    s = silhouette_samples(combinedData,cId) 
    sSum[ii-2] = sum(s) 
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) 
    plt.tight_layout() 

#Run this individually after above! 
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

#

numClusters = 4
kMeans = KMeans(n_clusters = numClusters).fit(combinedData) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

# determining the cluster siuation
ZeroList = []
OneList = []
TwoList = []
ThreeList = []

#Get indexes for eachart in each cluster
for i in range(len(cId)):
    if cId[i] == 0:
        ZeroList.append(i)
    elif cId[i] == 1:
        OneList.append(i)
    elif cId[i] == 2:
        TwoList.append(i)
    elif cId[i] == 3:
        ThreeList.append(i)

#Here I inspected per  parameter
# Zero List: Abstract Expressionism 3 |5/ 1 almost all|
# One List: Neoclassical 2 | 5/ 3 almost all |
# Two List: Rococo 2 | 5/ 2 almost all |
# Three List: Abstract almost all | 5/ Fairly Even, all abstract |

zExam = []
oExam = []
twExam = []
thExam = []


for i in ZeroList:
    zExam.append(theArt.iloc[i,5])
for i in OneList:
    oExam.append(theArt.iloc[i,5])
for i in TwoList:
    twExam.append(theArt.iloc[i,5])
for i in ThreeList:
    thExam.append(theArt.iloc[i,5])

print(statistics.mode(zExam))
print(statistics.mode(oExam))
print(statistics.mode(twExam))
print(statistics.mode(thExam))

'''
# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(combinedData.iloc[plotIndex,0],combinedData.iloc[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Challenges')
    plt.ylabel('Support')
'''

#%% Code for number 8

options = theData.iloc[:,205:215]

optionsAndRatings = pd.concat([options,pd.DataFrame(meanRatingsList)], axis = 1)

ratingsdata = theData.iloc[:,:91]
#For testing purposes 
optionsAndRatings2 = pd.concat([options,ratingsdata],axis=1)

cleanedOptionsAndRatings2 = optionsAndRatings2.dropna()

cleanedOptionsAndRatings = optionsAndRatings.dropna()

cleanedOptions = cleanedOptionsAndRatings.iloc[:,:10]
meanRatings8 = cleanedOptionsAndRatings.iloc[:,10]
medianRatings8 = cleanedOptionsAndRatings2.iloc[:,11]


zscoredData = stats.zscore(cleanedOptions)

pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_*-1

origDataNewCoordinates = pca.fit_transform(zscoredData)*-1

numPredictors = 10
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals)
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()
# Check which PC it is 

plt.subplot(1,2,1) # Factor 1: 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:]) 
plt.title('Overall Self Image ')
# So we found that the first one is the one! so now we use it in linear regression
test = origDataNewCoordinates[:,0].reshape(286,1)

PCA1Train, PCA1Test, ratingsTraining3, ratingsTesting3 = train_test_split(origDataNewCoordinates[:,0],meanRatings8, test_size = .2, random_state = 12844336)

view = pd.DataFrame(PCA1Train)


number8Trained = LinearRegression().fit(pd.DataFrame(PCA1Train),y = ratingsTraining3)

predictions3 = number8Trained.predict(pd.DataFrame(PCA1Test))
print("R^2 within model", number8Trained.score(pd.DataFrame(PCA1Train),ratingsTraining3))
print("RMSE for predictions", mean_squared_error(ratingsTesting3,predictions3, squared = False))
print("R^2 for predictions", metrics.r2_score(ratingsTesting3,predictions3))

#%% Code for number 9

options2 = theData.iloc[:,182:194]

options2AndRatings = pd.concat([options2,pd.DataFrame(meanRatingsList)],axis =1)

cleanedOptions2andRatings = options2AndRatings.dropna()

cleanedOptions2 = cleanedOptions2andRatings.iloc[:,:12]
cleanedRatings2 = cleanedOptions2andRatings.iloc[:,12]


zscoredData2 = stats.zscore(cleanedOptions2)
pca2 = PCA().fit(zscoredData2)
eigVals2 = pca2.explained_variance_
loadings2 = pca2.components_*-1

origDataNewCoordinates = pca2.fit_transform(zscoredData2)*-1

numPredictors = 12
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals2)
plt.title('Scree plot 2')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

# Check what we should name them  

#Factor 1: Overall, how good of a person are you? 
plt.subplot(1,2,1)  
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings2[0,:]) 
plt.title("Weights with PCA 1")
plt.show()
#Factor 2: How much attention do you need
plt.subplot(1,2,1) 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings2[1,:]) 
plt.title("Weights with PCA 2")

plt.show()
#Factor 3: How indifferent are you to others' feelings 
plt.subplot(1,2,1) # Factor 1: 
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings2[2,:]) 
plt.title("Weights with PCA 3")
plt.show()

newCoordData = origDataNewCoordinates[:,:3]
PCA1data = origDataNewCoordinates[:,0]
PCA2data = origDataNewCoordinates[:,1]
PCA3data = origDataNewCoordinates[:,2]

# ALL 3 together!
PCAall3Train, PCAall3Test, ratings4train, ratings4test = train_test_split(newCoordData, cleanedRatings2, test_size = .2, random_state = 12844336)

number9trained = LinearRegression().fit(PCAall3Train, y= ratings4train)

predictions4 = number9trained.predict(PCAall3Test)

print("(ALL 3) R^2 within model", number9trained.score(PCAall3Train,ratings4train))
print("(ALL 3) RMSE for predictions", mean_squared_error(ratings4test,predictions4, squared = False))
print("(ALL 3) R^2 for predictions", metrics.r2_score(ratings4test,predictions4))
print(number9trained.coef_)
print()

PCAall3withC = sm.add_constant(PCAall3Train)
seeingTheCoeffpvals = sm.OLS(ratings4train,PCAall3withC).fit()
seeingTheCoeffpvals.summary()


#%% Code for question 10

#Step One: always PCA 

# We need to get all info we have for each person somehow, we can do this by concatonating the response information to the avg ratings information and demographic info without the political party involveed 

FinalPredictors1 = pd.concat([theData.iloc[:,182:217], theData.iloc[:,218:]], axis = 1) # Without the ratings data just in case we want to use that 

FinalPredictors2 = pd.concat([FinalPredictors1,pd.DataFrame(meanEnergyList),pd.DataFrame(meanRatingsList)], axis =1)

FinalPredictors2andPolitics = pd.concat([FinalPredictors2, theData.iloc[:,217]], axis =1)
FinalPredictors2andPoliticsCleaned = FinalPredictors2andPolitics.dropna() # Drop the NA columns 

FinalPredictors2Cleaned = FinalPredictors2andPoliticsCleaned.iloc[:,:40]
PoliticsCleaned = FinalPredictors2andPoliticsCleaned.iloc[:,40]

#Finally, we have to convert the politics data to left(0) and not left (1)
FinalPoliticsList = []
for i in range(len(PoliticsCleaned)):
    if PoliticsCleaned.iloc[i] < 3:
        FinalPoliticsList.append(0)
    else:
        FinalPoliticsList.append(1)
FinalPoliticsDataFrame = pd.DataFrame(FinalPoliticsList)

#Now onto the PCA 
zscoredData3 = stats.zscore(FinalPredictors2Cleaned)
pca3 = PCA().fit(zscoredData3)
eigVals3 = pca3.explained_variance_
loadings3 = pca3.components_*-1

origDataNewCoordinates = pca3.fit_transform(zscoredData3)*-1

numPredictors = 40
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals3)
plt.title('Scree plot 3')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

#Elbow method is too hard to determine here, so we're going to have to use kaizer method (The ones have to take into account above 90%)
totalEigen = sum(eigVals3)
testSum = 0
ratio = 0
for i in range(len(eigVals3)):
    testSum += eigVals3[i]
    ratio = testSum/totalEigen
    if(ratio >= .9):
        print(i)
        break
# the result is 28... will not inspect for each but we can later algorithmically to see which is the most valid

relevantPredictors = origDataNewCoordinates[:,:29]

#We do a train test split

RFTrain, RFTest, PoliTrain, PoliTest = train_test_split(relevantPredictors, FinalPoliticsDataFrame, test_size = .2, random_state = 12844336)

numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(RFTrain,PoliTrain) #bagging numTrees trees

# Use model to make predictions:
predictions8 = clf.predict(RFTest) 

# Assess model accuracy:
modelAccuracy = accuracy_score(PoliTest,predictions8)
modelAUCscore = metrics.roc_auc_score(PoliTest,predictions8)
print('Random forest model accuracy:',modelAccuracy)
print('metrics.roc_auc_score:',modelAUCscore)

#%% Extra testing SIGNIFICANTLY WORSE 
FinalPredictors3 = pd.concat([theData.iloc[:,:217],theData.iloc[:,218:]], axis = 1)

FinalPredictors3Cleaned = FinalPredictors3.dropna()

zscoredData4 = stats.zscore(FinalPredictors3Cleaned)
pca4 = PCA().fit(FinalPredictors3Cleaned)
eigVals4 = pca4.explained_variance_
loadings4 = pca4.components_*-1

origDataNewCoordinates = pca4.fit_transform(zscoredData4)*-1

numPredictors = 220
plt.bar(np.linspace(1,numPredictors,numPredictors),eigVals4)
plt.title('Scree plot 3')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

#Elbow method is too hard to determine here, so we're going to have to use kaizer method (The ones have to take into account above 90%)
totalEigen = sum(eigVals4)
testSum = 0
ratio = 0
for i in range(len(eigVals4)):
    testSum += eigVals4[i]
    ratio = testSum/totalEigen
    if(ratio >= .9):
        print(i)
        break
relevantPredictors2 = origDataNewCoordinates[:,:103]
RFTrain2, RFTest2, PoliTrain2, PoliTest2 = train_test_split(relevantPredictors2, FinalPoliticsDataFrame, test_size = .2, random_state = 12844336)

numTrees = 100
clf2 = RandomForestClassifier(n_estimators = numTrees).fit(RFTrain2,PoliTrain2)

predictions9 = clf2.predict(RFTest2)

modelAccuracy = accuracy_score(PoliTest2,predictions9)
modelAUCscore = metrics.roc_auc_score(PoliTest2,predictions9)
print('Random forest model accuracy:',modelAccuracy)
print('metrics.roc_auc_score:',modelAUCscore)

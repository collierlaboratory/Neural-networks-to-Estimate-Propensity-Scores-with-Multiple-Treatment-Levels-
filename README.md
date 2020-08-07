# Neural-networks-to-Estimate-Propensity-Scores-with-Multiple-Treatment-Levels-
From "A Tutorial on Artificial Neural Networks in Propensity Score Analysis" in the Journal of Experimental Education (Collier & Leite, in press). 

# Python Code ##################

#Import training data ##################

import numpy as np #import library for arrays
data = ('train.csv')
import pandas as pd
#comma delimited is the default
data = pd.read_csv(data, header = 0)
#Convert panda to numpy array
data=data.values

#Seperate Covariates from Treatment Levels ##################

input_values = data[:, :-1] # gather covariates from random draws

target_values = data[:, [76]] ##gather true classes/treatments from the random draws


#Set the Hyperparameters ##################

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
       
#Train the Neural Network to the Data ################## 

 clf.fit(input_values, target_values) 
 
 
 #Import Validation Data ################## 
 
import numpy as np #import library for arrays

data = ('validation.csv')

import pandas as pd

#comma delimited is the default

data = pd.read_csv(data, header = 0)

#Convert panda to numpy array

data=data.values
 
 
 #Seperate Validation Covariates from Validation Treatment Levels ##################
 
input_values = data[:, :-1] # gather covariates from random draws

target_values = data[:, [76]] ##gather true classes/treatments from the random draws


 #Estimate Genearlized Propensity Scores using Validation Data ##################

from sklearn.metrics import log_loss

test_preds = clf.predict_proba(input_values)

cross_entropy = log_loss(target_values,test_preds) # calculate cross_entropy

print ('test stats')
print (cross_entropy)

#Import test data ##################

import numpy as np #import library for arrays

data = ('test.csv')

import pandas as pd

#comma delimited is the default

data = pd.read_csv(data, header = 0)

#Convert panda to numpy array

data=data.values

 #Seperate Test Covariates from Test Treatment Levels ##################
 
input_values = data[:, :-1] # gather covariates from random draws

target_values = data[:, [76]] ##gather true classes/treatments from the random draws

 
 #Test the Neural Network on Test Data ################## 
 
 from sklearn.metrics import log_loss

test_preds = clf.predict_proba(input_values)

cross_entropy = log_loss(target_values,test_preds) # calculate cross_entropy

print ('test stats')
print (cross_entropy)

 #Save the estimated Genearlized Propensity Scores ################## 
 
np.savetxt("GPS.csv", test_preds, delimiter=",")

# R Code ##################

#Now Read in data from tested neural network in R ##################

nn_ps <- read.csv("GPS.csv", header = F)

load("Chapter6_example_SASS_TFS_data_imputed.Rdata")

test <- imputedData

names(nn_ps)[1] <- "noMentor"

names(nn_ps)[2] <- "sameArea"

names(nn_ps)[3] <- "otherArea"

nn_ps <- cbind(nn_ps, imputedData)

#Obtain Inverse Probability Treatment Weights for Propensity score weighting

nn_ps$IPTW <- ifelse(nn_ps$Treat=="noMentor", 1/nn_ps$noMentor, 
                           ifelse(nn_ps$Treat=="sameArea", 1/nn_ps$sameArea, 1/nn_ps$otherArea))
with(nn_ps, by(IPTW,Treat,summary))


#Clip IPTW at the third quantile
#Extract rows by condition of treatment

noMentor <- nn_ps[nn_ps$Treat== "noMentor",]

sameArea <- nn_ps[nn_ps$Treat== "sameArea",]

otherArea <- nn_ps[nn_ps$Treat== "otherArea",]

#Clip IPTW at the third quantile by condition of treatment

noMentor$IPTW <- ifelse(noMentor$IPTW > quantile(noMentor$IPTW)[4],
                        quantile(noMentor$IPTW)[4], noMentor$IPTW)

sameArea$IPTW <- ifelse(sameArea$IPTW > quantile(sameArea$IPTW)[4],
                        quantile(sameArea$IPTW)[4], sameArea$IPTW)

otherArea$IPTW <- ifelse(otherArea$IPTW > quantile(otherArea$IPTW)[4],
                         quantile(otherArea$IPTW)[4], otherArea$IPTW)

#collaspe the treat conditions datasets into a single dataframe

data <- rbind(noMentor, sameArea);data <- rbind(data, otherArea)

nn_ps <- data

#Obtain the final weight that is the multiplication of IPTW and sampling weight TFNLWGT
nn_ps$IPTW.TFNLWGT <- with(nn_ps,IPTW*TFNLWGT)

with(nn_ps, by(IPTW.TFNLWGT,Treat,summary))

#normalize weights (make them sum to sample size)

nn_ps$finalWeightATE <- with(nn_ps,IPTW.TFNLWGT/
                                     mean(IPTW.TFNLWGT))
                                     
#check distribution of weights

with(nn_ps, by(finalWeightATE,Treat,summary))

#=============================================================
#Estimate ATE with propensity score weighting
#define the design
#school ids are provided to control for effects of clustering

require(survey)

#set up the survey design

design.IPTW <- svydesign(id = ~SCHCNTL, weights = ~finalWeightATE, data = nn_ps)

#Estimate treatmetn effects with regression

model.IPTW <- svyglm("leftTeaching~Treat",design=design.IPTW,
                     family="quasibinomial")
                     
summary(model.IPTW)




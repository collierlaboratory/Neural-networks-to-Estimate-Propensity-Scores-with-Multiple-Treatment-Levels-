\section{Code for Artificial Neural Networks in Propensity Score Analysis}

\subsection{The 80/20 Rule}

The R code for splitting the datafile “imputedData” is below:

\begin{lstlisting}[language=R]
# Sample 80% of the Data as Sample from Total 'n' Rows # of the Data 
sample <- sample.int(n = nrow(imputedData), size =floor(.80*
nrow(imputedData)), replace = F) 
train <- imputedData[sample, ]

# Extract the Remaining 20% of the Data
remain.20  <- imputedData[-sample, ]

# Extract Half of the Data for Validation (10%) 
sample <- sample.int(n = nrow(remain.20), size = floor(.50*
nrow(remain.20)), replace = F) 
validation <- remain.20[sample, ] 

# Extract the Remaining 10% for Testing
test <- remain.20[-sample, ]


\end{lstlisting}

We trained, validated, and tested our neural networks in Python. So, we saved our data splits to comma space-delimited files.


\begin{lstlisting}[language=R]
# Write CSV Files for Python 
write.csv(train, file = "train.csv", row.names = F)
write.csv(validation, file = "validation.csv", row.names = F) 
write.csv(test, file = "test.csv", row.names = F)

\end{lstlisting}


\subsection{Training Neural Networks}

\begin{lstlisting}[language=Python]

# Set Hyperparameters in Python
clf = MLPClassifier(activation='relu', alpha=1e- 05, 
hidden_layer_sizes=(15,), learning_rate='constant', learning_rate_init=
0.001, max_iter=200, solver='lbfgs')


\end{lstlisting}

The function MLPClassifier, implements a multi-layer perceptron (MLP) algorithm, and activation='relu' specifies the ReLU activation function,  which removes negative numbers and hold inputs ≥ 0 constant. The alpha parameter is a regularization term that reduces the variance of the model, hidden\_layer\_sizes specifies 15 neurons in two hidden layers, learning\_rate at constant and learning\_rate\_init at 0.001 controls the rate at which weights update, max\_iter stops the NN from learning at 100 iterations to prevent overfitting, and solver specifies the weight optimizer, lbfgs (Limited memory Broyden-Fletcher-Goldfarb-Shanno; Byrd, Nocedal, \& Schnabel, 1994), which converges faster on small datasets. 


Below is code showing how we applied the fit function to train the NN, “clf.”:  

\begin{lstlisting}[language={Python}]

# Train the Neural Network to the Data 
clf.fit(input_values, target_values)

\end{lstlisting}

In the code chunk above, the period in clf.fit is an operator that allows access to functions (e.g., fit) within the NN, “clf”, input\_values is a vector of cofounders, and target\_values  is a vector of treatment assignment. Next, the predict function estimates GPS using validation data, and the log\_loss function calculates cross-entropy:

\begin{lstlisting}[language=Python]

# Estimate Genearlized Propensity Scores on Validation Data 

from sklearn.metrics import log_loss 
test_preds = clf.predict_proba(input_values)


\end{lstlisting}

The test\_preds object contains a n x 3 matrix where each row is a participant and each column is the predicted probability of the participants to receive one of the three treatment levels.

\begin{lstlisting}[language=Python]

# Calculate cross-entropy for Validation Data 
cross_entropy = log_loss(target_values,test_preds) 
print (cross_entropy)

\end{lstlisting}


The code below is the same as the validation code, but uses the test data:

\begin{lstlisting}[language=Python]

# Estimate Genearlized Propensity Scores on Test Data 
test_preds = clf.predict_proba(input_values) 

# Calculate cross-entropy for Test Data 
cross_entropy = log_loss(target_values,test_preds) 
print (cross_entropy)

\end{lstlisting}

After finalizing our models, we estimated GPS for the entire sample: 

\begin{lstlisting}[language=Python]
# Estimation GPS for the Entire Sample 
entire_sample_preds = clf.predict_proba(input_values) 


\end{lstlisting}

In the code chunk above, the object entire\_sample\_preds is a vector of probabilities of treatment assignment (i.e., GPS).

The following code saves the GPS into a comma-seperated values file using the savetxt function in the numpy library:


\begin{lstlisting}[language=Python]

# Save the Estimated Genearlized Propensity Scores  
np.savetxt("GPS.csv", entire_sample_preds, delimiter=",")

\end{lstlisting}


\subsection{Evaluation of Common Support}

We employed R’s ggplot2 package to construct each set of box plots:

\begin{lstlisting}[language=R]

# Reshape the data 

data = reshape(data.frame(ps1,test[,c("CNTLNUM","Treat")]),
idvar="CNTLNUM",varying=c("noMentor","sameArea ","otherArea"), 
v.names="ps",times=c("GPS of No Mentor",
"GPS of Same Area","GPS of Other Area"),direction="long")  

require(ggplot2) 
ggplot(data, aes(x=time, y=ps, fill=Treat)) + geom_boxplot() +    facet_wrap(~Treat)+   scale_fill_grey()

\end{lstlisting}


\subsection{Calculation of Stabilized Inverse Probability of Treatment Weights}

Below is our SIPTW calculation in R using the GPS obtained with NN:

\begin{lstlisting}[language=R]

# Obtain Stabilized Inverse Probability Treatment Weights

imputedData$IPTW <- ifelse(imputedData$Treat=="noMentor", 
as.numeric(table(imputedData$Treat)[1])/nn_ps$noMentor, 
ifelse(imputedData$Treat=="sameArea", as.numeric(table(
imputedData$Treat)[2])/nn_ps$sameArea, as.numeric(table(
imputedData$Treat)[3])/nn_ps$otherArea))

with(imputedData, by(IPTW,Treat,summary))

\end{lstlisting}

We trimmed extreme weights downward at the top 5\% (Lee, Lessler, \& Stuart, 2011):


\begin{lstlisting}[language=R]

# Trim the Extreme Weights  
Q <- quantile(nn_ps$IPTW, probs=c(0, .95), na.rm = FALSE) 
iqr <- IQR(nn_ps$IPTW) 
nn_ps <- subset(nn_ps, nn_ps$IPTW > (Q[1] - 1.5*iqr) & nn_ps$IPTW < 
(Q[2]+1.5*iqr)) 

with(nn_ps, by(IPTW,Treat,summary))

\end{lstlisting}

\subsection{Average Treatment Effect Estimation}

We weighted cases using the SIPTWs in the final regression analysis using R’s survey package (Lumley, 2004):

\begin{lstlisting}[language=R]

# Call the survey package in R 
require(survey) 

# Set-up the survey design 
design.IPTW <- svydesign(id = ~SCHCNTL, weights = ~ IPTW, data = nn_ps)

\end{lstlisting}

The code above identifies cluster ids (i.e., variable SCHCNTL, which are school id numbers), the variable containing the SIPTW (i.e., finalWeightATE), and the dataset (i.e., nn\_ps).  


\begin{lstlisting}[language=R]



# Estimate treatment effects with regression 
model.IPTW <- svyglm("leftTeaching~as.factor(Treat)+URBANIC+
MINENR+MINTCH+PUPILS+T0106+T0125+T0154+
T0297+T0309+T0311+T0312+T0326+
T0331+schoolImputed" ,design=design.IPTW, family="quasibinomial")

summary(model.IPTW)


#Estimate treatment effects with regression

model.IPTW  <-  svyglm("leftTeaching~as.factor(Treat)+  
URBANIC+MINENR+MINTCH+PUPILS+T0106+T0125+T0154+T0297+T0309+T0311+T0312+
T0326+T0331+schoolImputed, 
design=design.IPTW, family="quasibinomial")

summary(model.IPTW)

\end{lstlisting}


In the code above, the outcome (ATE) model is defined as: 
\begin{lstlisting}[language=R]
leftTeaching ~ Treat1(sameMentor) + covariates.

\end{lstlisting}

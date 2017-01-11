require(rpart)
require(e1071) 
require(neuralnet)
require(RSNNS)

breastCancerDataWithNA<-read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=FALSE,sep=",")
#Omit the data with NA values
breastCancerData<-na.omit(breastCancerDataWithNA)

#Convert the class( M malignant and B Benign) into malignant as 1 and Benign as 0 
Malign=ifelse(breastCancerData$V2=="M",1,0)
breastCancerData=data.frame(breastCancerData,Malign)
breastCancerData=breastCancerData[-2]

#Rename the attribute names in the data set 
names(breastCancerData)<-c("ID","RV1","RV2","RV3","RV4","RV5","RV6","RV7","RV8","RV9","RV10","RV11","RV12",
"RV13","RV14","RV15","RV16","RV17","RV18","RV19",
"RV20","RV21","RV22","RV23","RV24","RV25","RV26","RV27","RV28","RV29","RV30","Diagnosis")

print("Decision trees")
#Decision Tree
#---------------
for(i in 1:5)
{
# Separate 80% of the data into test and remaining 20% to test data
sampleInstances<-sample(1:nrow(breastCancerData),size = 0.8*nrow(breastCancerData))
trainingData<-breastCancerData[sampleInstances,]
testData<-breastCancerData[-sampleInstances,]
 
#Create a decision tree model
DecisionTree.model<-rpart(as.factor(Diagnosis)~.,data=trainingData,
method='class',parms=list(split='information'),minsplit=2,minbucket=1)

#text(DecisionTree.model,pretty=0)
#cv_tree=cv.tree(DecisionTree.model,FUN=prune.misclass)
#test_prediction=predict(DecisionTree.model,testData,type ="class")
#mean(test_prediction != test_malign)
#plot(cv_tree$size,cv_tree$dev)
# Plot the tree size vs deviation. Deviation is minimum at tree size=6 for the given data

#Prune tree
cpVal<- DecisionTree.model$cptable[which.min(DecisionTree.model$cptable[,"xerror"]),"CP"]
prune_model=prune(DecisionTree.model,best=6,cp=cpVal)


#Predict the class for the test data 
test_prediction=predict(DecisionTree.model,testData,type = "class")
#Print the comparision of value predicted by the model and the actual class and calculate the accuracy
table(test_prediction,testData$Diagnosis)
accuracy <- sum(testData$Diagnosis == test_prediction)/nrow(testData)
print(accuracy)
}
print("Perceptrons")
#Perceptrons
#-------------
for(i in 1:5)
{
 
  #Splitting the data
  instances<-sample(1:nrow(breastCancerData),size = 0.8*nrow(breastCancerData))
  
  trainingData<-breastCancerData[instances,]
  testData<-breastCancerData[-instances,]
 
  
  #Building the model using training data
  bdValues <- trainingData[,1:31]
  bdTargets <- decodeClassLabels(trainingData[,32])
 
  #irisTargets <- decodeClassLabels(iris[,5], valTrue=0.9, valFalse=0.1)
  bd <- splitForTrainingAndTest(bdValues, bdTargets, ratio=0.15)
  bd <- normTrainingAndTestSet(bd)
 
  model <- mlp(bd$inputsTrain, bd$targetsTrain, size=0,inputsTest=bd$inputsTest, targetsTest=bd$targetsTest)
 
  val<- predict(model,bd$inputsTest)
  #Accuracy
  
  accuracy <- sum(round(val)==testData$Diagnosis)/nrow(testData)
  print(accuracy)
  
  
}
print("Neural Net")
#Neural Net 
#-------------

for(i in 1:5)
{
  n <- names(breastCancerData)
  f <- as.formula(paste("Diagnosis ~", paste(n[!n %in% "Diagnosis"], collapse = " + ")))
  #Splitting the data
  maxs =	apply(breastCancerData,	MARGIN	=	2,	max)
  mins =	apply(breastCancerData,	MARGIN	=	2,	min)
  scaled =	 as.data.frame(scale(breastCancerData,	center	=	mins,	scale	=	maxs	- mins))
  trainIndex	<- sample(1:nrow(scaled),	0.8 *	nrow(scaled))
  trainingData	<- scaled[trainIndex,	]
  testData	<- scaled[-trainIndex,	]
  scaled = as.data.frame(scale(breastCancerData, center = mins, scale = maxs - mins))
  
  #Building the model using training data
  nn <- neuralnet(f,data=trainingData,hidden=7,rep=3, threshold = 0.1,linear.output=T) 
  
  #Predicting using test data
  pred <- compute(nn,testData[,1:31])
  
  #Accuracy
  accuracy<- sum(testData$Diagnosis ==round(pred$net.result) )/nrow(testData)
  
 #accuracyArray <- rbind(accuracyArray,c(i,accuracy*100))
  print(accuracy)
  
}

#SVM
#--------
print("SVM")
  for(i in 1:5)
  {
    #Splitting the data
    instances<-sample(1:nrow(breastCancerData),size = 0.8*nrow(breastCancerData))
    
    trainingData<-breastCancerData[instances,]
    testData<-breastCancerData[-instances,]
    
    #Building the model using training data
    svmmodel<-svm(f,data=trainingData,kernel="radial")
    
    #Predicting using test data
    val<-predict(svmmodel,testData)
    
    #Accuracy
    accuracy<- sum(testData$Diagnosis == round(val))/nrow(testData)
    print(accuracy)
    
  }
  
#NaiveBayes
print("NaiveBayes")
#n <- names(breastCancerData)
#f <- as.formula(paste("Diagnosis ~", paste(n[!n %in% "Diagnosis"], collapse = " + ")))
for(i in 1:5)
{
  #Splitting the data
  instances<-sample(1:nrow(breastCancerData),size = 0.8*nrow(breastCancerData))
  
  trainingData<-breastCancerData[instances,]
  testData<-breastCancerData[-instances,]
  
  #Building the model using training data
  NBmodel<-naiveBayes(as.factor(Diagnosis)~.,data=trainingData)
  
  #Predicting using test data
  val<-predict(NBmodel,testData)
  
  #Accuracy
  accuracy<- sum(testData$Diagnosis == val)/nrow(testData)
  print(accuracy)
}

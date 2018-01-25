# setwd("/Users/gregfaletto/Documents/R/Fragile Families Challenge")

# rm(list=ls())   
# rm(list=ls()[!ls() %in% c("raw.train", "raw.background")])
# rm(list=ls()[!ls() %in% c("raw.train", "raw.background", "background",
# 	"m1", "x")])
# rm(list=ls()[!ls() %in% c("train", "bg", "model.e", "model.gpa",
	# "model.grit", "model.h", "model.j", "model.l")])

############# Load Libraries #################

# library(Amelia)
library(lattice)
library(ggplot2)
library(leaps)
library(caret)
library(pls)
library(RANN)
library(plyr)
library(gpls)
library(DMwR)
# library(penalized)
library(elasticnet)

################ Parameters ################

indicator <- 0 # used for diagnostic
# directory where output should be stored
dir.out <- "/Users/gregfaletto/Documents/R/Fragile Families Challenge/Predictions"

# directory where R file, raw data files live
dir.main <- "/Users/gregfaletto/Documents/R/Fragile Families Challenge"

cv.sets <- 45 # number of cross validation sets
pp <- c("nzv", "zv") # pre-processing parameters
cv.repeats <- 1 #number of times cross-validation should repeat

max.preds <- 2 # maximum number of predictors in subset selection models
prin.comps <- 3 # number of principal components for classifiers.

######### Pre-Pre-processing parameters

MS.th <- 0.25 # maximum proportion of observations that can be missing for a feature
LV.th <- 12 # maximum number of levels for a factor variable
max.cat <- 16 # maximum number of distinct integer values a feature can have
				# before I believe it's actually an int, not categorical

################ Continuous Model Parameters ################

# cont.con <- trainControl(method="cv", number=cv.sets)
cont.con <- trainControl(method="repeatedcv", number=cv.sets,
	repeats=cv.repeats, verboseIter=T, preProcOptions=list(verbose=T))
	# preProcOptions=list(verbose=T, freqCut=19/1, uniqueCut=10,
	# cutoff=0.75))
# myContGrid <- expand.grid(nvmax=max.preds)
# trControl=cont.con,
# myContGrid <- expand.grid(ncomp=c(1:15, 3*(6:17)))
myContGrid <- expand.grid(ncomp=c(1:35))
myContGrid.gpa <- expand.grid(ncomp=c(1:25))

################ Binary Model Parameters ################

cont.bin <- trainControl(method="cv", number=cv.sets,
	classProbs=T, verboseIter=T,
	preProcOptions=list(verbose=T))
# cont.bin <- trainControl(method=“cv”, number=cv.sets,
# 		summaryFunction=brier.func, classProbs=T)
myBinGrid <- expand.grid(ncomp=c(5*(1:6)))

################ Lasso Model Parameters ################

cont.lasso <- trainControl(method="repeatedcv", number=cv.sets,
	repeats=cv.repeats, verboseIter=T,
	preProcOptions=list(verbose=T))
# cont.bin <- trainControl(method=“cv”, number=cv.sets,
# 		summaryFunction=brier.func, classProbs=T)
myLassoGrid <- expand.grid(lambda1=c(10^-3, 10^-2, 10^-1, 10^0, 10^1,
	10^2, 10^3), lambda2=0)

################ Elastic Net Lasso Model Parameters ################

cont.ENet <- trainControl(verboseIter=T,
	preProcOptions=list(verbose=T))
# cont.bin <- trainControl(method=“cv”, number=cv.sets,
# 		summaryFunction=brier.func, classProbs=T)
myENetGrid <- expand.grid(fraction=(1/20)*(0:20))


############# Load Data ####################

## Read in training and background data, the latter may take a few minutes to run
print("Loading data...")
raw.train <- read.csv(file="train.csv",head=TRUE,sep=",")
raw.background <- read.csv(file="background.csv",head=TRUE,sep=",")
print("Data loaded!")

############### Putting it together ###############

train <- raw.train
background <- raw.background

################ Pre-pre-processing (lol) ################

######## Removing columns with one value and creating binary columns #####

  print("Removing columns with one value...")

  all_na <- logical(ncol(background))
  
  # Remove cols that have all missing vals 
  for (i in 1:length(all_na)){
    all_na[i] <- all(sapply(background[, i], is.na))
  }
  
  # # For each col...
  for (i in 2:ncol(background)) {
    vals <- unique(background[, i])
  #   # Edit entries with 1 val. or NA to binary data
  #   if (length(vals)==2) {
  #     # Find value that is not NA 
  #     if (is.na(vals[1])) {
  #       v <- vals[2]
  #     } else {
  #       v <- vals[1]
  #     }
  #     # Set all NA to FALSE and all values to TRUE 
  #     background[, i][background[, i]!=v] <- FALSE
  #     background[, i][background[, i]==v] <- TRUE
  #   } 
    # Remove cols. with only one value 
    if (length(vals)==1) {
      all_na[i] <- TRUE
    }
}
  
  background[, all_na] <- NULL
# }

print(paste("Removed", sum(as.integer(all_na)), "features."))

if(is.null(background$challengeID)){
  print("Oops you messed up and deleted challngeID on this step.")
}

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

############### Factorize Non-numerical Features ######

print("Factorizing non-numerical and integer features...")

counter <- 0

for(i in 2:ncol(background)){
  if( (!is.numeric(background[, i]) ) |
  	( (is.integer(background[, i]))&(length(unique(background[, i]))<max.cat) ) ){
  		## Change negative values to NA (seemed to make things worse)
  		# neg.indices <- sign(background[, i]==-1)
  		# background[, i][neg.indices] <- NA
  		# Count as missing any values with invalid skips
  		background[, i][background[, i]==-9|background[, i]==-8|
  		background[, i]==-5|
  		background[, i]==-3|background[, i]==-2|background[, i]==-1] <- NA
  		background[, i] <- factor(background[, i])
  		counter <- counter+1
  	}
  }

for(i in 2:ncol(background)){
  if(is.integer(background[, i])){
    background[, i] <- factor(background[, i])
    counter <- counter+1
  }
}

print(paste("Factorized", counter, "non-numerical and integer features."))

if(is.null(background$challengeID)){
  print("Oops you messed up and deleted challngeID on this step.")
}

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

# # ####### Delete those nominal variables with more than threshold number of levels #

# # # nominal.test <- !logical(ncol(background))
# # # for(i in 2:ncol(background)){
# # #   if(is.factor(background[, i])){
# # #     nominal.test[i] <- (length(levels(background[, i]) <= LV.th))
# # #   }
# # # }

# # # background <- background[, nominal.test]

# # # print(paste("Eliminated", sum(!nominal.test), "categorical variables with", 
# # #   "more than", LV.th, "levels."))

# # # if(is.null(background$challengeID)){
# # #   print("Oops you messed up and deleted challngeID on this step.")
# # # }

# # # if(!is.null(background$challengeID.1) & indicator==0){
# # # 	print("Something weird happened on this step and you created all those weird challengeID variables.")
# # # 	indicator <- 1
# # # }

######### Eliminate Features with more than threshold of data missing #######

print(paste("Eliminating features with more than", MS.th,
  "of observations missing"))

missing.obs <- apply(is.na(background), 2, mean)
background <- background[, missing.obs < MS.th]

print(paste("Removed", sum(as.integer(missing.obs)), "features."))

if(is.null(background$challengeID)) {
  print("Oops you messed up and deleted challngeID on this step.")
}

if(!is.null(raw.train$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

###### kNN imputation

print("Imputing missing values using kNN imputation...")

background <- knnImputation(background, k=5)

print("Values imputed!")

###### LAST STEP: removing columns with emtpy values (bad idea?) ####

print("Removing features with empty values...")

no.missing <- complete.cases(t(background))
background <- background[, no.missing]

print(paste("Removed", sum(as.integer(!no.missing)),
	"features with empty values."))

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

####################### Centering Data ################

print(paste("Centering and standardizing data..."))

for(i in (2:ncol(background))) {
	if(is.numeric(background[, i])) {
		background[, i] <- (background[, i] - mean(background[, i])/
			sd(background[, i]))
	}
}

print("Data centered and standardized!")

if(is.null(background$challengeID)){
  print("Oops you messed up and deleted challngeID on this step.")
}

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

######## Removing columns with one value and creating binary columns #####

  print("Removing columns with one value...")

  all_na <- logical(ncol(background))
  
  # Remove cols that have all missing vals 
  for (i in 1:length(all_na)){
    all_na[i] <- all(sapply(background[, i], is.na))
  }
  
  # # For each col...
  for (i in 2:ncol(background)) {
    vals <- unique(background[, i])
    # Remove cols. with only one value 
    if (length(vals)==1) {
      all_na[i] <- TRUE
    }
}
  
  background[, all_na] <- NULL

print(paste("Removed", sum(as.integer(all_na)), "features."))

if(is.null(background$challengeID)){
  print("Oops you messed up and deleted challngeID on this step.")
}

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}


########### Putting it together ##################

# merge together a working file of training data with background data
m1 <- merge(background, train, by="challengeID", all=FALSE)

x <- m1[, !(colnames(m1) %in% c("challengeID", "gpa", "grit",
	"materialHardship", "eviction", "layoff", "jobTraining"))]

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

### Factorize binary outcomes

m1$eviction <- as.factor(m1$eviction)
m1$layoff <- as.factor(m1$layoff)
m1$jobTraining <- as.factor(m1$jobTraining)

m1$eviction <- mapvalues(m1$eviction, from = c("0", "1"), to = c("No", "Yes"))
m1$layoff <- mapvalues(m1$layoff, from = c("0", "1"), to = c("No", "Yes"))
m1$jobTraining <- mapvalues(m1$jobTraining, from = c("0", "1"),
	to = c("No", "Yes"))

if(!is.null(background$challengeID.1) & indicator==0){
	print("Something weird happened on this step and you created all those weird challengeID variables.")
	indicator <- 1
}

################# Variables of interest for lasso model ############

lbw <- "cm1lbw"
cog.edu <- c("cm3cogsc", "cf3cogsc", "cm1edu", "cf1edu")
income <- c("cm1hhinc", "cf1hhinc", "cm2hhinc", "cf2hhinc", "cm3hhinc",
	"cf3hhinc", "cm4hhinc", "cf4hhinc", "cm1inpov", "cm2povco", 
	"cm3povco", "cm4povco", "cm1povca", "cm2povca", "cm3povca",
	"cm4povca", "cf1inpov", "cf2povco", "cf3povco", "cf4povco",
	"cf1povca", "cf2povca", "cf3povca", "cf4povca", "cf2povcob",
	"cf3povcob", "cf4povcob", "cf2povcab", "cf3povcab", "cf4povcab")
jail <- c("cf1finjail", "cf2finjail", "cf3finjail", "cf4finjail", "cf5finjail",
	"cmf1finjail", "cmf2finjail", "cmf3finjail", "cmf4finjail",
	"cmf5finjail", "cf1fevjail", "cf2fevjail", "cf3fevjail", "cf4fevjail",
	"cf5fevjail", "cmf1fevjail", "cmf2fevjail", "cmf3fevjail", "cmf4fevjail",
	"cmf5fevjail")
men.health <- c("cm3alc_case", "cm3drug_case", "cm1gad_case", "cm2gad_case",
	"cm3gad_case", "cm4gad_case", "cm5gad_case", "cm1md_case_con",
	"cm2md_case_con", "cm3md_case_con", "cm4md_case_con", "cm5md_case_con",
	"cm1md_case_lib", "cm2md_case_lib", "cm3md_case_lib", "cm4md_case_lib",
	"cm5md_case_lib", "cf3alc_case", "cf3drug_case", "cf1gad_case",
	"cf2gad_case", "cf3gad_case", "cf4gad_case", "cf5gad_case", 
	"cf1md_case_con", "cf2md_case_con", "cf3md_case_con", "cf4md_case_con",
	"cf5md_case_con", "cf1md_case_lib", "cf2md_case_lib", "cf3md_case_lib",
	"cf4md_case_lib", "cf5md_case_lib")

relationships <- c("cm1relf", "cm2relf", "cm3relf", "cm4relf", 
	"cm1marf", "cm2marf", "cm3marf", "cm4marf", "cm2amrf", "cm3amrf",
	"cm4amrf", "cm1cohf", "cm2cohf", "cm3cohf", "cm4cohf", "cm2alvf",
	"cm3alvf", "cm4alvf", "cm2finst", "cm2stflg", "cf1marm", "cf2marm",
	"cf3marm", "cf4marm", "cf1cohm", "cf2cohm" , "cf3cohm", "cf4cohm")

partner <- c("cm2marp", "cm3marp", "cm4marp", "cm2cohp", "cm3cohp",
	"cm4cohp", "cf2marp", "cf3marp", "cf4marp", "cf2cohp", "cf3cohp",
	"cf4cohp")

others <- c("cm1ethrace", "cf1ethrace")

vars <- c(lbw, cog.edu, income, jail, men.health, partner, others)
# dat.lasso <- data.frame(rep(0, nrow(x)))
vars.in.lasso <- c()

for (i in 1:length(vars)){
	if (!is.null(x[[vars[i]]])){
		vars.in.lasso <- c(vars.in.lasso, vars[i])
		# dat.lasso <- data.frame(dat.lasso, x[[vars[i]]])
	}
}

### dat.lasso <- dat.lasso[, 2:ncol(dat.lasso)]
### dat.lasso <- x[, vars.in.lasso]

################ Continuous Models (Subset Selection) ################
#
#
#
#




################ Training Continuous Models ################

# gpa

print("Training GPA model...")
results.gpa <- train(x=x[!is.na(m1$gpa), ], y=m1$gpa[!is.na(m1$gpa)],
	method="pcr", metric="RMSE", preProcess=pp, trControl=cont.con,
	tuneGrid=myContGrid, na.action=na.omit)
# results <- train(x=x, y=m1$gpa, method="leapForward", metric="RMSE",
# 	preProcess=pp, trControl=cont.con, tuneGrid=myContGrid, na.action=na.omit)
print("GPA model done!")
print("GPA model:")
print(summary(results.gpa$finalModel))

save(results.gpa, file="gpa_pcr_model.R")
## load(file="gpa_pcr_model.R")

rmse.pcr.gpa <- min(results.gpa$results$RMSE)

print("Training GPA lasso model...")
x.gpa <- x[!is.na(m1$gpa), vars.in.lasso]
gpa <- m1$gpa[!is.na(m1$gpa)]
lm.gpa <- data.frame(x.gpa, gpa)
# results.lm.gpa <- lm(gpa~., data=lm.gpa)
# results.lasso.gpa <- train(x=x.gpa, y=y.gpa, method="penalized", metric="RMSE",
# 	preProcess=pp, trControl=cont.lasso, tuneGrid=myLassoGrid)
# results.lasso.gpa <- train(x=x.gpa, y=y.gpa, method="lasso", metric="RMSE",
# 	preProcess=pp, trControl=cont.ENet, tuneGrid=myENetGrid)
results.lasso.gpa <- train(gpa~., data=lm.gpa, method="lasso",
	preProcess=pp, trControl=cont.ENet)
# rmse.lm.gpa <- 1/(length(results.lm.gpa$residuals))*sum((results.lm.gpa$residuals)^2)
rmse.lasso.gpa <- min(results.lasso.gpa$results[, "RMSE"])
print("GPA lasso model done!")
# print("GPA lasso model:")
# print(summary(results.lasso.gpa$finalModel))

save(results.lasso.gpa, file="gpa_lasso_model.R")
##load(file="gpa_lasso_model.R")

# # create predictions on the background dataset
MyData <- background[, "challengeID", drop=F]
gpa.model.desc <- ""

if(rmse.lasso.gpa < rmse.pcr.gpa){
	MyData$gpa <- predict(results.lasso.gpa, newdata=background[, vars.in.lasso])
	gpa.model.desc <- paste("Lasso-penalized linear regression with",
		length(vars.in.lasso), "variables.")
	
} else {
	MyData$gpa <- predict.train(results.gpa, newdata=background)
	gpa.model.desc <- paste("Principal components linear regression with",
		as.character(results.gpa$bestTune), "components.")
	
}

MyData$gpa[is.na(MyData$gpa)] <- 2.866738197
MyData$gpa[MyData$gpa < 1] <- 1
MyData$gpa[MyData$gpa > 4] <- 4

rm(results.lasso.gpa)
rm(results.gpa)


# # # grit

print("Training grit model...")
results.grit <- train(x=x[!is.na(m1$grit), ], y=m1$grit[!is.na(m1$grit)],
	method="pcr", metric="RMSE", preProcess=pp, trControl=cont.con,
	tuneGrid=myContGrid, na.action=na.omit)
print("Grit model done!")
print("Grit model:")
print(summary(results.grit$finalModel))
save(results.grit, file="grit_pcr_model.R")

rmse.pcr.grit <- min(results.grit$results$RMSE)

print("Training grit lasso model...")
x.grit <- x[!is.na(m1$grit), vars.in.lasso]
grit <- m1$grit[!is.na(m1$grit)]
lm.grit <- data.frame(x.grit, grit)
results.lasso.grit <- train(grit~., data=lm.grit, method="lasso",
	preProcess=pp, trControl=cont.ENet)
rmse.lasso.grit <- min(results.lasso.grit$results[, "RMSE"])
print("Grit lasso model done!")
save(results.lasso.grit, file="grit_lasso_model.R")

# # create predictions on the background dataset
grit.model.desc <- ""

if(rmse.lasso.grit < rmse.pcr.grit){
	MyData$grit <- predict(results.lasso.grit, newdata=background[, vars.in.lasso])
	grit.model.desc <- paste("Lasso-penalized linear regression with",
		length(vars.in.lasso), "variables.")
	
} else {
	MyData$grit <- predict.train(results.grit, newdata=background)
	grit.model.desc <- paste("Principal components linear regression with",
		as.character(results.grit$bestTune), "components.")
	
}

MyData$grit[is.na(MyData$grit)] <- 3.427538787
MyData$grit[MyData$grit < 1] <- 1
MyData$grit[MyData$grit > 4] <- 4

rm(results.lasso.grit)
rm(results.grit)

# # # # materialHardship

print("Training material hardship model...")

# library("rms")
# model.h <- orm()
x.mh <- x[!is.na(m1$materialHardship), ]
y.mh <- m1$materialHardship[!is.na(m1$materialHardship)]
results.mh <- train(x=x.mh, y=y.mh, method="pcr", metric="RMSE",
	preProcess=pp, trControl=cont.con, tuneGrid=myContGrid, na.action=na.omit)
# results.mh <- train(x=x.mh, y=y.mh, method="pls", metric="brier", maximize=F,
# 	preProcess=pp, trControl=cont.bin, tuneGrid=myBinGrid)
# model.h <- glm(formula=materialHardship ~., family=binomial(link="logit"),
# 	data=m1[, !(colnames(m1) %in% c("gpa", "grit", "eviction", "layoff",
		# "jobTraining"))])
print("Material hardship model done!")
print("Material hardship model:")
print(summary(results.mh$finalModel))

save(results.mh, file="mh_pcr_model.R")

rmse.pcr.mh <- min(results.mh$results$RMSE)

print("Training material hardship lasso model...")
x.mh <- x[!is.na(m1$materialHardship), vars.in.lasso]
materialHardship <- m1$materialHardship[!is.na(m1$materialHardship)]
lm.mh <- data.frame(x.mh, materialHardship)
results.lasso.mh <- train(materialHardship~., data=lm.mh, method="lasso",
	preProcess=pp, trControl=cont.ENet)
rmse.lasso.mh <- min(results.lasso.mh$results[, "RMSE"])
print("Material hardship lasso model done!")

save(results.lasso.mh, file="mh_lasso_model.R")

# # create predictions on the background dataset
mh.model.desc <- ""

if(rmse.lasso.mh < rmse.pcr.mh){
	MyData$materialHardship <- predict(results.lasso.mh,
		newdata=background[, vars.in.lasso])
	mh.model.desc <- paste("Lasso-penalized linear regression with",
		length(vars.in.lasso), "variables.")
	
} else {
	MyData$materialHardship <- predict.train(results.mh, newdata=background)
	mh.model.desc <- paste("Principal components linear regression with",
		as.character(results.mh$bestTune),
		"components.")
	
}

MyData$materialHardship[is.na(MyData$materialHardship)] <- 0.103744782
MyData$materialHardship[MyData$materialHardship < 0] <- 0
MyData$materialHardship[MyData$materialHardship > 1] <- 1

rm(results.lasso.mh)
rm(results.mh)

# ############### Training Binary Models ################

# # # # eviction
# # x.e <- x[!is.na(m1$eviction), ]
# # y.e <- m1$eviction[!is.na(m1$eviction)]
# # results.e <- train(x=x.e, y=y.e, method="pls",
# # 	trControl=cont.bin, tuneGrid=myBinGrid)
# # # results.e <- train(x=x.r, y=y.r, method="pls", metric="brier", maximize=F,
# # # 	preProcess=pp, trControl=cont.bin, tuneGrid=myBinGrid)
# # print("Eviction model done!")
# # print("Eviction model:")
# # print(summary(results.e$finalModel))

# # # # layoff
# # x.l <- x[!is.na(m1$layoff), ]
# # y.l <- m1$layoff[!is.na(m1$layoff)]
# # results.l <- train(x=x.l, y=y.l, method="pls", metric="brier", maximize=F,
# # 	preProcess=pp, trControl=cont.bin, tuneGrid=myBinGrid)
# # # results.l <- train(x=x.l, y=y.l, method="pls", metric="brier", maximize=F,
# # # 	preProcess=pp, trControl=cont.bin, tuneGrid=myBinGrid)
# # # model.l <-  glm(formula=layoff ~. -challengeID, family=binomial(link="logit"), data=m1[,
# # # 	!(colnames(m1) %in% c("gpa", "grit", "materialHardship", "eviction",
# # # 		"jobTraining"))])
# # print("Layoff model done!")
# # print("Layoff model:")
# # print(summary(results.l$finalModel))

# # # # job training
# # x.j <- x[!is.na(m1$jobTraining), ]
# # y.j <- m1$obTraining[!is.na(m1$jobTraining)]
# # results.j <- train(x=x.j, y=y.j, method="pls", metric="brier", maximize=F,
# # 	preProcess=pp, trControl=cont.bin, tuneGrid=myBinGrid)
# # # results.j <- train(x=x.j, y=y.j, method="pls", metric="brier", maximize=F,
# # # 	preProcess=pp, trControl=cont.bin, tuneGrid=myBinGrid)
# # # model.j <- glm(formula=jobTraining ~. -challengeID, family=binomial(link="logit"), data=m1[,
# # # 	!(colnames(m1) %in% c("gpa", "grit", "materialHardship", "eviction",
# # # 		"layoff"))])
# # print("Job training model done!")
# # print("Job training model:")
# # print(summary(results.j$finalModel))



# # MyData$gpa <- rep(2.866738197, nrow(background))
# # MyData$materialHardship <- predict.train(results.mh, newdata=background)
# # # clarify want probabilites for the logits, not classification prediction
# # MyData$materialHardship <- predict.train(results.mh, newdata=background)
# # MyData$eviction <- predict.train(results.e, newdata=background, type="prob")
# # MyData$layoff <- predict.train(results.l, newdata=background, type="prob")
# # MyData$jobTraining <- predict.train(results.j, newdata=background, type="prob")

# # # Extract just those variables needed for the submission
# # # MyData<-subset(background,select=c(challengeID,gpa,grit,materialHardship,eviction,layoff,jobTraining))

# # # patch any missing values left, with the mean, to avoid a submission failure

# # MyData$eviction[is.na(MyData$eviction)] <- 0.059629883
# # MyData$layoff[is.na(MyData$layoff)] <- 0.20908379
# # MyData$jobTraining[is.na(MyData$jobTraining)] <- 0.234770705
# # MyData$gpa <- rep(2.866738197, nrow(background))
# # MyData$grit <- rep(3.427538787, length(MyData$gpa))
# # MyData$materialHardship <- rep(0.103744782, length(MyData$gpa))



MyData$eviction <- rep(0.059629883, length(MyData$gpa))
MyData$layoff <- rep(0.20908379, length(MyData$gpa))
MyData$jobTraining <- rep(0.234770705, length(MyData$gpa))

# # Ensure no remaining NAs, and sensible-looking results
print("")
print("Summary of data:")
print(summary(MyData))

# Output to prediction file
setwd(dir.out)
write.csv(MyData, file="prediction.csv",row.names=FALSE)

# Write out the narrative file too
fileConn <- file("narrative.txt")
writeLines(c("GPA model:", gpa.model.desc, "Grit model:", grit.model.desc,
			"Material hardship model:", mh.model.desc,
             "Thank you to Stephen McKay AKA the_Brit, whose code 'FFC-simple-R-code.R' I",
             "used to get started, Viola Mocz (vmocz) & Sonia Hashim (shashim), whose code",
             "FeatEngineering.R I used, and hty, whose code 'COS424_HW2_imputation_Rcode.R' I",
             "used. Those files were taken from the Fragile Familes github."),
fileConn)
close(fileConn)

setwd(dir.main)
# required libraries
library(caret)
library(Boruta)

# THIS SCRIPTS REMOVES DUPLICATED VARIABLES AND VARIABLES WITH ZERO VARIANCE, AND SAVES NEW DATASET
# AS TRAIN_CLEAN AND TEST_CLEAN
# THEN IT IDENTIFIES IMPORTANT FEATURES USING BORUTA PACKAGE AND SAVES THEM IN CSV


# load train and test set
train = read.csv('train.csv')
test = read.csv('test.csv')

# Remove ID columns
train$ID = NULL
test_ID = test$ID
test$ID = NULL

# Remove target variable
train_target = train$TARGET
train$TARGET = NULL

# Add column indicating how many 0 values per row
count0 = function(x) { return(sum(x==0)) }

train$no0 = apply(train, 1, FUN=count0)
test$no0 = apply(test,1,FUN=count0)

dim(train);dim(test)


# Remove constant features
for (feat in (names(train))){
    if (length(unique(train[[feat]]))==1){
        train[[feat]] = NULL
        test[[feat]] = NULL
    }
}

dim(train);dim(test)


# Remove identical features
feat_pairs = combn(names(train), 2, simplify = F)
toRemove = c()
for (pair in feat_pairs){
    f1 = pair[1]
    f2 = pair[2]
    
    if ( !(f1 %in% toRemove) & !(f2 %in% toRemove) ){
        if (all(train[[f1]] == train[[f2]])) {
            toRemove = c(toRemove, f2)
        }
    }
}

toRemove

featToKeep= setdiff(names(train), toRemove)

train = train[,featToKeep]
test = test[,featToKeep]

dim(train); dim(test)

write.csv(train, "train_clean.csv", row.names = F)
write.csv(test, "test_clean.csv", row.names = F)

#=============== Select features with Boruta package

train$target = train_target

# select random sample for analysis using caret createDataPartition() function
set.seed(123)
idx <- createDataPartition(train$target, p=0.03, list=FALSE)
train_small <- train[idx,]

# get names of feature variables
features <- setdiff(names(train_small),c("target"))

dim(train); dim(train_small)

set.seed(13)
bor.results <- Boruta(train_small[,features], factor(train_small$target), maxRuns=101,doTrace=0)

#plot results
plot(bor.results)

# make dataframe
bor_res = as.data.frame(bor.results$finalDecision)
names(bor_res) = 'decision'

#selected features
sel_feat = subset(bor_res, decision == 'Confirmed')
sel_feat

write.csv(sel_feat, "boruta_features_03_perc.csv", row.names = F)

set.seed(123)
idx <- createDataPartition(train$target, p=0.06, list=FALSE)
train_small <- train[idx,]

# get names of feature variables
features <- setdiff(names(train_small),c("target"))

dim(train); dim(train_small)

set.seed(13)
bor.results2 <- Boruta(train_small[,features], factor(train_small$target), maxRuns=101,doTrace=0)

#plot results
plot(bor.results2)

# make dataframe
bor_res2 = as.data.frame(bor.results2$finalDecision)
names(bor_res2) = 'decision'

#selected features
sel_feat2 = subset(bor_res2, decision == 'Confirmed')
sel_feat2

write.csv(sel_feat2, "boruta_features_06_perc.csv", row.names = F)


library(lubridate)
library(skimr)
library(dplyr)
library(DataExplorer)
library(stringr)
library(mltools)
library(data.table)
library(factoextra)
library(ggplot2)
library(e1071)
library(caret)
library(MLmetrics)
library(randomForest)
library(qcc)
library(xgboost)


#######################################################################################
#################################### EXPLORATION #########################################
#######################################################################################

#Reading the dataset
data <- read.csv(file='violin (1).csv')
data$Birth.date..Marker. <-as.integer(substr(gsub("[^0-9]", "", data$Birth.date..Marker.),1,4))
data$USD <- as.numeric(data$USD)

# summary of the new data
summary(data)


# Creating a pareto chart to get the top auction houses and what share of data they contain
counts <- table(data$Auction.House)
pareto.chart(counts)


# Creating a pareto chart to get the top makers and what share of data they contain
counts <- table(data$maker)
pareto.chart(counts)


## Question: Does the auction house affect the price of the violin bows?
## Answer: It does. The median values (shown below) indicate that auction houses fetch different prices

# Found the median price of the 10 auction houses to find whether there is a relative 
#co-relation between Auction Houses and Price

tarisio = data[data$Auction.House == 'Tarisio',]
median(tarisio$USD)

sotheby = data[data$Auction.House == 'Sotheby\'s',]
median(sotheby$USD)

Christie = data[data$Auction.House == 'Christie\'s',]
median(Christie$USD)

Phillip  = data[data$Auction.House == 'Phillip\'s',]
median(Phillip$USD)

Skinner  = data[data$Auction.House == 'Skinner',]
median(Skinner$USD)

Bongartz  = data[data$Auction.House == 'Bongartz\'s',]
median(Bongartz$USD)

Bonhams  = data[data$Auction.House == 'Bonhams',]
median(Bonhams$USD)

Vichy = data[data$Auction.House == 'Vichy-Enchères',]
median(Vichy$USD)

Brompton  = data[data$Auction.House == "Brompton's",]
median(Brompton$USD)

Ingles = data[data$Auction.House == 'Ingles & Hayday',]
median(Ingles$USD)


# Identifying highest selling Maker

maker<-data %>%
  group_by(maker) %>%
  summarise(price =sum(USD)) %>%
  ungroup %>%
  mutate(total = sum(price), price_percentage = 100*(price / total)) %>%
  arrange(desc(price))

maker<- maker[-3]
maker


#Identifying highest maker and auction house

Maker_Auction<-data %>%
  group_by(Auction.House,maker) %>%
  summarise(price =sum(USD)) %>%
  ungroup %>%
  mutate(total = sum(price), price_percentage = 100*(price / total)) %>%
  arrange(desc(price))

Maker_Auction<- Maker_Auction[-3]
Maker_Auction


# identifying maker's contribution in an auction house
data_table=table(data$maker,data$Auction.House)

for (i in unique(data$Auction.House)){
  Maker_dist<-data %>%
    filter(data$Auction.House ==i) %>%
    group_by(maker) %>%
    summarise(price =sum(USD)) %>%
    ungroup %>%
    mutate(total = sum(price), price_percentage = 100*(price / total)) %>%
    arrange(desc(price))
  
  for(j in unique(Maker_dist$maker)){
    Maker_dist[which(Maker_dist$maker==j),4]
    rnum=which(row.names(data_table)==j)
    colnum=which(colnames(data_table)==i)
    data_table[rnum,colnum]=as.numeric(Maker_dist[which(Maker_dist$maker==j),4])
  }
  
}

data_table

#######################################################################################
#################################### MODELING #########################################
#######################################################################################

#This function is used to create the preprocessed dataset with the non-grouping
#More detailed explanations about the grouped and non-grouped dataset is in the documentation

data_preprocessing_ungrouped <- function(data, variables_to_model_on, sale_date_preprocessing){
  
  #Converting the USD to numeric
  data$USD <- as.numeric(str_replace_all(data$USD, "[^[:alnum:]]", ""))
  
  #Adding model with sale date, but trying both - by modifying the date to numeric and keeping as is
  
  if(sale_date_preprocessing == 'numeric'){
    print("Converting sale date to numeric")
    data$Sale.Date <- as.Date(data$Sale.Date)
    data$Sale.Date <- as.numeric(data$Sale.Date)
    data$Sale.Date <- as.factor(data$Sale.Date)
  }

  #Selecting 3 columns for modelling
  data<-data[, variables_to_model_on]
  
  #Converting to factors 
  data$Auction.House <- as.factor(data$Auction.House)
  data$maker <- as.factor(data$maker)
  
  
  # One hot encoding of Auction house and maker
  new_data<- one_hot(as.data.table(data))
  names(new_data) <- make.names(names(new_data), unique=TRUE)
  
  #Returning the preprocessed dataset
  return(new_data)
}



#This function creates the preprocessed dataset but after grouping and taking median
#More details are explained in the document shared with this code

data_preprocessing_grouped <- function(data, sale_date_include) {
  
  #Converting the USD to numeric
  data$USD <- as.numeric(str_replace_all(data$USD, "[^[:alnum:]]", ""))
  
  if(sale_date_include == 'include'){
    print("Including Sale date in our model")
    #Creating the group by function for maker, auction house and city combinations
    data<-data %>%
      group_by(Auction.House,maker,City,Sale.Date) %>%
      summarise(USD =mean(USD)) 
    
    #Converting them to factors
    data$Auction.House <- as.factor(data$Auction.House)
    data$maker <- as.factor(data$maker)
    data$City <- as.factor(data$City)
    data$Sale.Date <- as.factor(data$Sale.Date)  }
  else {
    #Creating the group by function for maker, auction house and city combinations
    data<-data %>%
      group_by(Auction.House,maker,City) %>%
      summarise(USD =mean(USD)) 
    
    #Converting them to factors
    data$Auction.House <- as.factor(data$Auction.House)
    data$maker <- as.factor(data$maker)
    data$City <- as.factor(data$City)
      }


  # One hot encoding of 
  new_data<- one_hot(as.data.table(data))
  names(new_data) <- make.names(names(new_data), unique=TRUE)
  
  #Returning the dataset
  return(new_data)
  
  
}


#Creating a linear regression function since this will be called multiple times
#This function returns the summary of the model. We specifically care about the R sq. values

linear_regression_model <- function(data_to_run) {
  #Linear Regression 
  lr_model<- lm(data_to_run$USD ~.,data_to_run)
  cur_summary = summary(lr_model)
  
  return(cur_summary)
}

#Creating a svm regression function
#This function returns the summary of the model. But this model does not work for our dataset

svm_regressor_model <- function(data_to_run) {
  
  # SVM Regressor 
  set.seed(123)
  #Creating the train and test set for the model to get metrics
  indexes = createDataPartition(data_to_run$USD, p = .8, list = F)
  train = data_to_run[indexes, ]
  test = data_to_run[-indexes, ]
  modelsvm = svm(train$USD ~.,data=train)
  
  #Predicting values
  pred = predict(modelsvm, test)
  mse = MSE(test$USD, pred)
  mae = MAE(test$USD, pred)
  rmse = RMSE(test$USD, pred)
  r2 = R2(test$USD, pred, form = "traditional")
  
  #Returning the summary
  cur_summary = cat(" MAE:", mae, "\n", "MSE:", mse, "\n", 
      "RMSE:", rmse, "\n", "R-squared:", r2)
  
  return(cur_summary) 
}


random_forest_regressor <- function(data_to_run, x_tree_depth) {
  #Tree size of 50 covers about 40% of the variance of the data
  set.seed(4543)
  rf.fit <- randomForest(USD ~ ., data=data_to_run, ntree=x_tree_depth,
                         keep.forest=FALSE, importance=TRUE)
  return(rf.fit)
}

#######################################################################################

# Using the dataset with limited auction houses (top 10) and limited makers (top 180)
data <- read.csv(file="select_auction_houses_select_makers.csv")

#Calling the preprocessing function but GROUPED by MEAN
processed_data = data_preprocessing_grouped(data, 'exclude')
length(processed_data)
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
svm_regressor_model(processed_data)

#Calling the preprocessing function but on UNGROUPED data
processed_data = data_preprocessing_ungrouped(data, c('Auction.House','maker','USD'), 'n/a')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
svm_regressor_model(processed_data)


#######################################################################################


# Using the dataset with all auction houses and limited makers (top 180)
data <- read.csv(file="select_makers_all_auction_houses.csv")

#Calling the preprocessing function but GROUPED by MEAN
processed_data = data_preprocessing_grouped(data, 'exclude')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
svm_regressor_model(processed_data)

#Calling the preprocessing function but on UNGROUPED data
processed_data = data_preprocessing_ungrouped(data, c('Auction.House','maker','USD'), 'n/a')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
svm_regressor_model(processed_data)



#######################################################################################


# Using the entire dataset
data <- read.csv(file ="violin (1).csv")

#Calling the preprocessing function but GROUPED by MEAN
processed_data = data_preprocessing_grouped(data, 'exclude')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
svm_regressor_model(processed_data)
#Running a random forest regressor on the model with tree = 1 explains over 50% of the variation in the data, making it a strong model
random_forest_regressor(processed_data, 1)

#Calling the preprocessing function but on UNGROUPED data
processed_data = data_preprocessing_ungrouped(data, c('Auction.House','maker','USD'), 'n/a')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
svm_regressor_model(processed_data)
#Running a random forest regressor on the model with tree = 1
random_forest_regressor(processed_data, 1)


###Code for running XGBoost
indexes = createDataPartition(processed_data$USD, p = .8, list = F)
#Creating train test split
train = processed_data[indexes, ]
train1 = train[,0:601]
train_x = data.matrix(train1)
train_y = train$USD

test = processed_data[-indexes, ]
test1 = test[,0:601]
test_x = data.matrix(test1)
test_y = test$USD

xgb_train = xgb.DMatrix(data = train_x, label = unlist(train_y))
xgb_test = xgb.DMatrix(data = test_x, label = unlist(test_y))

xgbc = xgboost(data = xgb_train, max.depth = 50, nrounds = 50)
pred_y = predict(xgbc, xgb_test)
#Getting R sq for the model
res <- caret::postResample(test_y, pred_y)
rsq <- res[2]
rsq


#######################################################################################
#######################################################################################
########## Modeling using sale date ##############
#######################################################################################
#######################################################################################

# Using the entire dataset
data <- read.csv(file="violin (1).csv")

#Adding the sale date to the input variables, and converting the sale date into NUMERIC
processed_data = data_preprocessing_ungrouped(data, c('Auction.House','maker','USD','Sale.Date'), 'numeric')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
#Running a random forest regressor on the model with tree = 1 explains over 50% of the variation in the data, making it a strong model
random_forest_regressor(processed_data, 1)
#The default kernel in SVM is RBF (radial basis), This has also turned out to give a negative r sq
#So, like you said, it does not look like SVM works well for categorical variables
svm_regressor_model(processed_data)
#Also checking random forest
random_forest_regressor(processed_data, 1)



#Adding the sale date to the input variables, and keeping the sale date as is without converting to numeric
processed_data = data_preprocessing_ungrouped(data, c('Auction.House','maker','USD','Sale.Date'), 'non-numeric')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
random_forest_regressor(processed_data, 1)


#Calling the preprocessing function but GROUPED, and INCLUDING Sale date
processed_data = data_preprocessing_grouped(data, 'include')
linear_reg_summary = linear_regression_model(processed_data)
linear_reg_summary
#Here also, running a random forest regressor on the model with tree = 1 explains over 50% of the variation in the data
random_forest_regressor(processed_data, 1)






### With using sale date ####

# Using the entire dataset
data <- read.csv(file= "violin (1).csv")

#Adding the sale date to the input variables, and converting the sale date into NUMERIC
processed_data = data_preprocessing_ungrouped(data, c('Auction.House','maker','Sale.Date','USD'), 'numeric')
indexes = createDataPartition(processed_data$USD, p = .8, list = F)

train = processed_data[indexes, ]
train1 = train[,0:1360]
train_x = data.matrix(train1)
train_y = train$USD

test = processed_data[-indexes, ]
test1 = test[,0:1360]
test_x = data.matrix(test1)
test_y = test$USD
xgb_train = xgb.DMatrix(data = train_x, label = unlist(train_y))
xgb_test = xgb.DMatrix(data = test_x, label = unlist(test_y))

xgbc = xgboost(data = xgb_train, max.depth = 50, nrounds = 20)
pred_y = predict(xgbc, xgb_test)

res <- caret::postResample(test_y, pred_y)
rsq <- res[2]
rsq




processed_data = data_preprocessing_grouped(data, 'include')


indexes = createDataPartition(processed_data$USD, p = .8, list = F)

train = processed_data[indexes, ]
train1 = train[,0:601]
train_x = data.matrix(train1)
train_y = train$USD

test = processed_data[-indexes, ]
test1 = test[,0:601]
test_x = data.matrix(test1)
test_y = test$USD
xgb_train = xgb.DMatrix(data = train_x, label = unlist(train_y))
xgb_test = xgb.DMatrix(data = test_x, label = unlist(test_y))

xgbc = xgboost(data = xgb_train, max.depth = 50, nrounds = 50)
pred_y = predict(xgbc, xgb_test)

res <- caret::postResample(test_y, pred_y)
rsq <- res[2]
rsq


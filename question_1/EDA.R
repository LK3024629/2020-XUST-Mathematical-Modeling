---
title: "EDA"
author: "LI KONG"
date: "5/13/2020"
---

library(tidyverse)
library(readxl)
library(mice)
library(VIM)
library(missForest)
library(lubridate)
library(grDevices)
library(EnvStats)

## Read data
original_data <- read_excel("Datasets/data.xls", na = "<NULL>")

##Get summary
summary(original_data)

##Number of data columns
num_row <- ncol(original_data)

## Removing last row from data frame
original_data <- original_data[, -num_row]

## Get summary
summary(original_data)

## returns number of missing values in each variable of a dataset
##sum(is.na(data frame$column name)
colSums(is.na(original_data))

## mice package has a function known as md.pattern(). It returns a tabular form of missing value present in each variable in a data set.
md.pattern(original_data)

## Convert data format to data frame 
original_data <- as.data.frame(original_data)

## Dataset with missing values
missing_data = original_data[c('T0215', 'T1145','T1700','T1745','T2115', 'T1430','T1715','T1730','T2400' )]

## Visualization of missing data
md.pattern(missing_data)
mice_plot <- aggr(missing_data, col=c('navyblue','yellow'),
                  prop=F, numbers=T, sortVars=TRUE,
                  labels=names(missing_data), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

## imputing missing value with mi
imputed <- missForest(original_data, verbose = TRUE)

#check imputed values
imputed_data_vis <- imputed$ximp

#check imputation error
estimation_error<- imputed$OOBerror

##Fill value visualization
data1 <- data.frame(x=ymd(imputed_data_vis$date), y=missing_data$T2400, lab="missing_data")
data2 <- data.frame(x=ymd(imputed_data_vis$date), y=imputed_data_vis$T2400, lab="imputed_data")
data3 <- rbind(data2,data1)
gg <- ggplot(data = data3,aes(x=x,y=y,color=lab)) + geom_point()
print(gg) 

## Box plot of site data
print(boxplot(imputed_data_vis[2:98]))

## Outlier detection
imputed_data <- imputed$ximp
name1 <- names(imputed_data[2:98])
print(name1)
for (i in name1){
   rosnerTest(imputed_data[,i], k = 6, warn = F)
   height <- imputed_data[,i]
   for(j in 1:length(height))
      {
      if(height[j] > quantile(height,0.75)+1.5*IQR(height))
        {
        ## Replace outliers with missing values
        imputed_data[,i][j] <- NA
        # imputed_data[,i][j] <- max(height[1:(i-1)])
        }
   }
   height <- 0
} 

## Box plot after replacing outliers with missing values
print(boxplot(imputed_data[2:98]))

## Get summary
summary(imputed_data)

abnormal_missing_data_vis <- imputed_data[,-1]
abnormal_missing_data <- imputed_data
## Visualization of abnormal missing data
md.pattern(abnormal_missing_data)
mice_plot <- aggr(abnormal_missing_data_vis, col=c('navyblue','yellow'),
                  prop=F, numbers=T, sortVars=TRUE,
                  labels=names(abnormal_missing_data_vis), cex.axis=.7,
                  gap=3, ylab=c("Abnormal missing data","Pattern"))

## imputing missing value with mi
abnormal_imputed <- missForest(abnormal_missing_data, verbose = TRUE)

#check abnormal_imputed values
abnormal_imputed_data <- abnormal_imputed$ximp

#check imputation error
abnormal_estimation_error<- abnormal_imputed$OOBerror

##Fill value visualization
data1 <- data.frame(x=ymd(abnormal_missing_data$date), y=abnormal_missing_data$T2400, lab="abnormal_missing_data")
data2 <- data.frame(x=ymd(abnormal_missing_data$date), y=abnormal_imputed_data$T2400, lab="abnormal_imputed")
data3 <- rbind(data2,data1)
gg <- ggplot(data = data3,aes(x=x,y=y,color=lab)) + geom_point()
print(gg) 

## Box plot of site data
print(boxplot(imputed_data_vis[2:98]))

## Box plot after replacing outliers with missing values
print(boxplot(imputed_data[2:98]))

## Box plot after replacing outliers with missforest values
print(boxplot(abnormal_imputed_data[2:98]))


write.csv(x = abnormal_imputed_data,file = 'Datasets/final_data.csv')




























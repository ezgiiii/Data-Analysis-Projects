# You can download diabetes.csv file from this web site:
# https://www.kaggle.com/datasets/saurabh00007/diabetescsv

import pandas as pd
from scipy.stats import shapiro, mannwhitneyu
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel

df_= pd.read_csv("diabetes.csv")
df=df_.copy()

# AB Teting: Is there a link between age and having diabetes?

df.groupby("Outcome").agg({"Age": "mean"})
# We observe a difference but is it statistically correct or just a coincidence?

# 1. Hypothesis
# H0: M1 = M2 (There is not defference)

# H1: M1 != M2 (There are differences)

# Is the outcome normally distributed?
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value is lower than 0.005 in both outcome.
# We should use non-parametric test. (mannwhitneyu)

test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value is lower than 0.005. Than we say "There are statistically difference between M1 and M2."
# So this means there is a link between Age and Outcome (having diabetes)

spark = SparkSession.builder.appName("spark").getOrCreate()

# Spark DataFrames are distrubuted collections of data into named columns
# the differens with pandas DataFrame is Spark DataFrames are immutable.
df = spark.read.csv('diabetes.csv',header=True,inferSchema=True)

# header=True is used to get the values as their data types
# inferSchema=True her column un data tipini otomatik olarak almaya yarar.

df.show()

df.printSchema() # data types of the column

print((df.count(),len(df.columns))) # total number of columns and rows

df.groupby('Outcome').count().show() # Total number of patient who are diabetic and non diabetic

df.describe().show()

# find null variables
for col in df.columns:
  print(col+":",df[df[col].isNull()].count())


# find total numbers of 0 values in Glucose, Bloodpressure, SkiinThickness, Insulin and BMI columns

def count_zeros():
  columns_list = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
  for i in columns_list:
    print(i+":" , df[df[i]==0].count())

count_zeros()

for i in df.columns[1:6]:
  data = df.agg({i:'mean'}).first()[0]
  print("Mean Value For {} is {} ".format(i,int(data)))
  df = df.withColumn(i,when(df[i]==0,int(data)).otherwise(df[i]))

# withColumn() method generates a new dataset with specified features

# See the correlations of each column with Outcome
for col in df.columns:
  print("correlation to outcome for {} is {}".format(col,df.stat.corr('Outcome',col)))


# #Vector assembler merges multiple column into single vector column
# we need to add 1 additional feature column obtaining all the information of the columns we want
# the machine learning algorithm to consider

assembler = VectorAssembler(inputCols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
                            outputCol='features')

# since we dont have highly correlated column we consider all the columns
# VectorAssembler takes a set of columns of dataset and define an additional column as features

output_data = assembler.transform(df)
# transform method creates a new column of 'features'

output_data.printSchema()

output_data.show()

#logistic regression is useful for binary classification column

final_data = output_data.select('features','Outcome')

final_data.printSchema()

train , test = final_data.randomSplit([0.7,0.3])
models = LogisticRegression(labelCol='Outcome')
model = models.fit(train)

summary = model.summary

summary.predictions.describe().show()

predictions = model.evaluate(test)

predictions.predictions.show(30)
#row prediction is the raw output of the logistic regression

#binary classification evaluater
evaluator= BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',labelCol='Outcome')


evaluator.evaluate(model.transform(test))
#it gives how accurate our model is

model.save("model")
# write.overwrite().save(path)

model = LogisticRegressionModel.load('model')


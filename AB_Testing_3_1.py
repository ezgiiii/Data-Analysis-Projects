# Story of the datasets:
# There is an online website which applies "maximum bidding" and "average bidding"
# Maximum bidding data collected in control group dataset and average bidding data is collected in test group dataset.
# We will determine which approach is better using AB Testing.

import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df_actual_control = pd.read_excel("ab_testing.xlsx", sheet_name="Control Group", usecols=["Impression","Click","Purchase","Earning"])
df_control= df_actual_control.copy()

df_actual_test = pd.read_excel("ab_testing.xlsx", sheet_name="Test Group", usecols=["Impression","Click","Purchase","Earning"])
df_test= df_actual_test.copy()

df_control.head()
df_control.describe().T

df_test.head()
df_test.describe().T

# We can observe that there are differences between test and control datasets
# But is this just a coincidence or statistically accurate?

# Hypothesis:
# H0: M1 = M2 There aren't any statistical difference between test and control datasets.

# H1: M1 != M2 There is a statistical difference between test and control datasets.

# First things first. We should check weather these data is normally distributed or not.
stat, pvalue_control = shapiro(df_control["Purchase"])
print("Control dataset Normality p-value: ",str(pvalue_control))

stat, pvalue_test = shapiro(df_test["Purchase"])
print("Test dataset Normality p-value: ",str(pvalue_test))

# p-values are higher than 0.005
# Both control and test set is normally distributed.
# Next, we should check the homogeneity of the variances

stat, pvalue = levene(df_control["Purchase"],df_test["Purchase"])
print("Homogeneity of variance p-value: ",str(pvalue))

# p-value also higher than 0.005.
# We should use parametric test

stat, pvalue = ttest_ind(df_control["Purchase"],df_test["Purchase"], equal_var=True)

print("p-value = %.4f" % pvalue)

# p-value is higher than 0.005
# So We can finally say that our H0 hypothesis is statistically accurate.
# There aren't any difference between control and test group.
# Then, we can't say that maximum bidding is any different than average bidding for this company.





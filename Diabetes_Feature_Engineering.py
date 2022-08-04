import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_=pd.read_csv("diabetes.csv")
df=df_.copy()

df.describe().T

df.info()
df.head()
df.isnull().sum() # There aren't any null values.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {cat_cols}')
    print(f'num_cols: {num_cols}')
    print(f'cat_but_car: {cat_but_car}')
    print(f'num_but_cat: {num_but_cat}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# We replace 0 values with NaN except Outcome and Pregnancy.
# Outcome and Pregnancy can be zero but the other values can not be zero.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

na_cols=missing_values_table(df,na_name=True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_cols)

# I want to predict the missing values
# dff=pd.get_dummies(df, drop_first=True)  every column has numeric values we don't need this code

# Standardizing the variables
# MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# df has scaled values. We will change them back to the way we understand.
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

for i in num_cols:
    sns.boxplot(x=df[i])
    plt.show()

def outlier_thresholds(dataframe, col_name, q1=0.5, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:
    print(col, check_outlier(df,col))


# Lets look at Local Outlier Factor to locate the outliers considering columns together first.
# Local Outlier Factor

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

# we decide a threshold value to choose outlier
# Elbow Method:
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# when we check out the score plot, we observe a significant change after 2nd variable.
# so we accept first two variable as outlier and remove them from the dataframe
threshold = np.sort(df_scores)[2]

df[df_scores < threshold].drop(axis=0, labels=df[df_scores < threshold].index)


# correlation heatmap
sns.heatmap(df.corr(),annot=True)
plt.show()

# According to heatmap Glucose and Insulin has the highest correlation with Outcome/Diabetes

for col in df.columns:
    print(col, check_outlier(df,col))
    
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


# Feature Extraction
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glucose 140 ın altı risk yok 140-200 diyabet riski var 200 ve üzeri diyabet
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# blood pressure 70'in altı düşük 70-80 normal 80 üstü büyük
df["BLOOD_PRESSURE"] = pd.cut(x= df["BloodPressure"], bins=[0,70,80,150], labels= ["Low", "Normal", "High"])

# Pregnancies
df["NEW_PREGNANCIES"] = pd.cut(x=df["Pregnancies"], bins= [-1,6,11,17], labels=["0_6","6_11","11_17"])

# Pregnancy does not have a direct correlation to diabetes but when a woman has a tendency to diabetes
# after pregnancy she has a higher chance to has diabetes.
# Lets declare NEW_GLUCOSE_PREGNANCY variable



df.loc[(df["Glucose"]<=140) & (df["Pregnancies"] <= 6) , "NEW_GLUCOSE_PREGNANCY"] = "normal0_6"
df.loc[((df["Glucose"]>140) & (df["Glucose"]<200)) & (df["Pregnancies"] <= 6) , "NEW_GLUCOSE_PREGNANCY"] = "Prediabetes0_6"
df.loc[(df["Glucose"]>=200) & (df["Pregnancies"] <= 6) , "NEW_GLUCOSE_PREGNANCY"] = "Diabetes0_6"

df.loc[(df["Glucose"]<=140) & ((df["Pregnancies"] > 6) & (df["Pregnancies"] <= 11)), "NEW_GLUCOSE_PREGNANCY"] = "normal6_11"
df.loc[((df["Glucose"]>140) & (df["Glucose"]<200)) & ((df["Pregnancies"] > 6) & (df["Pregnancies"] <= 11)) , "NEW_GLUCOSE_PREGNANCY"] = "Prediabetes6_11"
df.loc[(df["Glucose"]>=200) & ((df["Pregnancies"] > 6) & (df["Pregnancies"] <= 11)) , "NEW_GLUCOSE_PREGNANCY"] = "Diabetes6_11"

df.loc[(df["Glucose"]<=140) & ((df["Pregnancies"] > 11) & (df["Pregnancies"] <= 17)), "NEW_GLUCOSE_PREGNANCY"] = "normal11_17"
df.loc[((df["Glucose"]>140) & (df["Glucose"]<200)) & ((df["Pregnancies"] > 11) & (df["Pregnancies"] <= 17)) , "NEW_GLUCOSE_PREGNANCY"] = "Prediabetes11_17"
df.loc[(df["Glucose"]>=200) & ((df["Pregnancies"] > 11) & (df["Pregnancies"] <= 17)) , "NEW_GLUCOSE_PREGNANCY"] = "Diabetes11_17"

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df.columns = [col.upper() for col in df.columns]

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# LABEL ENCODING for numerical columns
df["OUTCOME"] = df["OUTCOME"].astype(str)
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

labelencoder = LabelEncoder()
labelencoder.fit_transform(df[binary_cols])


# One-Hot Encoding for categorical columns
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])



# MODELLING
y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)


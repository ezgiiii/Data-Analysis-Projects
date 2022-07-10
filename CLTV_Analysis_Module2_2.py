import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



dff=pd.read_csv("flo_data_20k.csv")
df=dff.copy()

# Separate the data into 3 quartiles.
# Normally 0.25,0.75 values are used but in this data frame the number of transaction and frequency values
# are not extremely high. If I use regular box plot method I should change a lot of variables
# I just want to get rid of some problematic values. I want to mask the problematic values.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Get the masked dataframe
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")
replace_with_thresholds(df,"customer_value_total_ever_offline")


df["total_shopping"] = df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"]+df["customer_value_total_ever_offline"]


date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)

# CLTV = BG/NBD Model * Gamma gamma submodel
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7
# RECENCY: The number of weeks passed after the last order

cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
# TANURE: The number of weeks passed after the first order

cltv_df["frequency"] = df["total_shopping"]
# FREQUENCY: The number of orders made

cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["total_shopping"]
# MONETARY: Average gain per order


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# expected sales in three months
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# expected sales in six months
cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])


cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

ggf = GammaGammaFitter(penalizer_coef=0.001)

cltv_df["exp_average_value"] = ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'], cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv


cltv_df.sort_values("cltv",ascending=False)[:20]


cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.groupby("cltv_segment").agg(["max","mean","count"])









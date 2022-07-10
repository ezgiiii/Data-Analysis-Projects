import datetime as dt
import pandas as pd

df_real = pd.read_csv("flo_data_20k.csv")
df=df_real.copy()

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df.head(10)
df.columns
desc=df.describe().T  # aykırı değer var
df.isnull().sum() #null değer yok

df.dtypes
df.info

df["total_shopping_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_spend"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

#converting dates to datetime
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# df["first_order_date"] = pd.to_datetime(df["first_order_date"], format='%Y-%m-%d')
# df["last_order_date"] = pd.to_datetime(df["last_order_date"], format='%Y-%m-%d')
# df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"], format='%Y-%m-%d')
# df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"], format='%Y-%m-%d')


df.groupby("order_channel").agg({"master_id":"count",
                                 "total_shopping_num":"sum",
                                 "total_spend":"sum"})

df["total_spend"].nlargest(n=10) # show 10 most spent customers
# df.sort_values("customer_value_total", ascending=False)[:10]

df["total_shopping_num"].nlargest(n=10) # show 10 most shopped customers
# df.sort_values("order_num_total", ascending=False)[:10]

df.dropna(inplace=True) #clear the na and nand values from the data set



# def dataPreperation (df):
#     #function for data prepping which has done above
#
#     df["total_shopping_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
#     df["total_spend"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
#     date_columns = df.columns[df.columns.str.contains("date")]
#     df[date_columns] = df[date_columns].apply(pd.to_datetime)
#     return df
#
# df = dataPreperation(df)


df["last_order_date"].max() #get the date for last order

#assign the analysis date to a random date after the last orders date
analysis_date = dt.datetime(2010, 6, 1)
# type(today_date)


# rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (analysis_date - x.max()),
#                                   "total_shopping_num": lambda x: x,
#                                   "total_spend": lambda x: x})
#
# rfm.columns = ['recency', 'frequency', 'monetary']
# rfm["recency"] = rfm["recency"].astype('timedelta64[D]')

#prep the rfm dataframe
rfm = pd.DataFrame()

rfm["customer_id"] = df["master_id"]

rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
#RECENCY: How recently a customer has made a purchase (days)

rfm["frequency"] = df["total_shopping_num"]
#FREQUENCY: How often a customer makes a purchase

rfm["monetary"] = df["total_spend"]
#MONETARY: How much money a customer spends on purchases

rfm.describe().T

#segmentations of RFM Values
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[5,4,3,2,1])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[5,4,3,2,1])

rfm["RF_Score"]=rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str)


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_Score'].replace(seg_map, regex=True)


rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["min","mean","max","count"])

#Say that a store adds a new product to its selection. The price tag is high so the store wants to do the advertisement
# only  to its frequent customers. These customers are "champions" and "loyal_customers" segments and women (kadın).

target_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("new_brand_target_customer_id.csv", index=False)
cust_ids.shape


#Say that the store wants to do a discount on male kids section and wants to advertise the "cant_loose" "hibernating"
# "new_customers" segments
discount=pd.DataFrame()
target_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
discount_ids = df[(df["master_id"].isin(target_customer_ids)) & ((df["interested_in_categories_12"]
                                                                  .str.contains("ERKEK"))|(df["interested_in_categories_12"]
                                                                                           .str.contains("COCUK")))]["master_id"]
discount_ids.to_csv("discount_target_customer_ids.csv", index=False)





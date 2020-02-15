#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:21:04 2020

@author: luismiguel
"""

############# SAME DATA PRE PROCESSING ###########


import pandas as pd
import datetime
from datetime import timedelta
import geopy.distance

df = pd.read_csv("/Users/luismiguel/Desktop/McGill MMA/Enterprise Analytics/accidents.csv")

df = df.drop("Unnamed: 0",axis=1)
start = df["Start_Time"].tolist()
end = df["End_Time"].tolist()
years = []
months =[]
days = []
hours =[]
ends = []
##Getting individual tokens to get the hour, month and  year
for time in start:
    years.append(time.split(" ")[0].split("-")[0])
    months.append(time.split(" ")[0].split("-")[1])
    days.append(time.split(" ")[0].split("-")[2])
    hours.append(time.split(" ")[1].split(":")[0])
    
datetimeFormat = '%Y-%m-%d %H:%M:%S'
for begin,finish in zip(start,end):
    ends.append((datetime.datetime.strptime(finish, datetimeFormat) - datetime.datetime.strptime(begin, datetimeFormat)).seconds)

df["Start_Year"] = years
df["Start_Month"] = months
df["Start_Day"] = days
df["Start_Hour"] = hours
df["Accident_Duration"] = ends
df = df.drop("Start_Time",axis=1)
df = df.drop("End_Time",axis=1)

df['End_Lng'] = df['End_Lng'].fillna(0)
df['End_Lat'] = df['End_Lat'].fillna(0)
start_lat = df["Start_Lat"].tolist()
start_long = df["Start_Lng"].tolist()
end_long = df["End_Lng"].tolist()
end_lat = df["End_Lat"].tolist()
distance = []
i = 0
for i in range(0,500000):
    if end_long[i] == 0:
        end_long[i] = start_long[i]
    if end_lat[i] == 0:
        end_lat[i] = start_lat[i]
    coord1 = (float(start_lat[i]),float(start_long[i]))
    coord2 =  (float(end_lat[i]),float(end_long[i]))
    distance.append(geopy.distance.vincenty(coord1, coord2).mi)

df["Distance(mi)"] = distance

df['Temperature(F)'] = df['Temperature(F)'].fillna(df['Temperature(F)'].median())
df['Wind_Chill(F)'] = df['Wind_Chill(F)'].fillna(df['Wind_Chill(F)'].median())
df['Humidity(%)'] = df['Humidity(%)'].fillna(df['Humidity(%)'].median())
df['Pressure(in)'] = df['Pressure(in)'].fillna(df['Pressure(in)'].median())
df['Visibility(mi)'] = df['Visibility(mi)'].fillna(df['Visibility(mi)'].median())
df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].median())
df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())
df = df.dropna()

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

#######################   FEATURE IMPORTANCE    #####################################################################



df_FI = pd.get_dummies(df, columns=['Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'])
df_FI = df_FI.sample(n=50000, random_state = 5)

##### RFE

X = df_FI.iloc[:,13:171]
y = df_FI["Severity"]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

Ir = LogisticRegression()
rfe = RFE(Ir,20)
model = rfe.fit(X,y)

RFE_features = pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking']).sort_values(by='ranking', ascending = True)

##### LASSSO

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df_FI.iloc[:,13:171]
y = df_FI["Severity"]
X_std = scaler.fit_transform(X)

from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01, positive=True)
model.fit(X_std, y)

LASSO_featues = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient']).sort_values(by='coefficient', ascending = False)

##### RANDOM FOREST

X = df_FI.iloc[:,13:171]
y = df_FI["Severity"]

from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0)

model = randomforest.fit(X,y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X,y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
RF_features = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values(by='Gini coefficient', ascending = False)
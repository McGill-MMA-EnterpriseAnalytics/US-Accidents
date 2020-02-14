# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:43:51 2020

@author: 小丁
"""

#import libraries
import pandas as pd
from dataclasses import dataclass
import numpy as np
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
h2o.init()
h2o.connect()
#upload dataset
df = pd.read_csv(r'C:\Users\小丁\OneDrive\桌面\accidents.csv')
train = h2o.import_file(pd.DataFrame(df[0:400000]))
test = h2o.import_file(pd.DataFrame(df[400000:500000]))

x = list(train.columns)
y = 'Severity'
x.remove(y)

aml = H2OAutoML(max_models = 20, seed = 1)
aml.train(x=x,y=y, training_frame = train)
lb = aml.leaderboard
lb  = get_leaderboard(aml, extra_columns = 'ALL')
lb.head(rows= lb.nrows)

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:59:06 2018

"""
import pandas as pd
import numpy as np
#import datetime as dt 
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor#, RandomForestRegressor, BaggingRegressor,AdaBoostRegressor
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from sklearn.model_selection import train_test_split
import os, sys
import yaml
import pyodbc
print(os.path.dirname(os.path.realpath(sys.argv[0])))
file_name = os.path.basename(sys.argv[0])
file_location = os.path.dirname(os.path.realpath(sys.argv[0]))
###############
from pytz import timezone
from datetime import datetime,timedelta
# to get the delta beteen 
def get_diff(now, tzname):
    tz = timezone(tzname)
    utc = timezone('UTC')
    utc.localize(datetime.now())
    delta =  utc.localize(now) - tz.localize(now)
    return delta

now = datetime.utcnow()
print(now)
tzname = 'Europe/Berlin'
delta = get_diff(now, tzname)
print(delta)
job_start_date0 = now + delta
#print(now_in_berlin)

current = job_start_date0.replace(microsecond=0,second=0,minute=0)+timedelta(hours=1)

job_start_date1 = str(job_start_date0.strftime('%Y-%m-%d %H:%M:%S'))
job_start_date = datetime.strptime(job_start_date1, '%Y-%m-%d %H:%M:%S')

LS = pd.read_csv(r'C:\Users\rtkri\Downloads\DataSheet.csv', encoding='utf-8-sig')
LS['Dates'] = pd.to_datetime(LS['Dates'])
LS.dropna(inplace=True)
LS.set_index('Dates', inplace=True)
####
'''
1st Two Letters: CL = Crude Oil
2nd Letter = Month of Delivery, e.g. – N = July
Final symbol: Number corresponding to the year, e.g. - 5
Thus, the October, 2016 contract for crude oil on the NYMEX is expressed  as “CLV6”.  
Other important symbols:  NG = Natural Gas, HO = Heating Oil,  RB = Unleaded gasoline, PN = Propane.
Months
F = January		K = May 	U = September
G = February 		M = June 	V = October
H = March 		N = July 	X = November
J =  April 			Q = August 	Z = December   
'''
import pandas as pd
import requests
import json
url="https://www.cmegroup.com/CmeWS/mvc/Quotes/FutureContracts/XNYM/G?quoteCodes=CLZ8"
s=requests.get(url).content
data = json.loads(s.decode('utf-8'))
from pandas.io.json import json_normalize
data_df = json_normalize(data['quotes'])
data_required = data_df[['change','last','priorSettle','open','close','high','low','quoteCode','volume','updated']]
data_required['updated'] = data_required['updated'].str.replace('<br />','')
data_required['updated'] = pd.to_datetime(data_required['updated'].str[12:]+' '+data_required['updated'].str[0:12])


###########################

import sys

# The wget module
#import wget

# The BeautifulSoup module
# The selenium module
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
#from selenium.webdriver.chrome.options import Options
import pandas as pd

#options = Options()
#options.add_argument("--headless")
#driver = webdriver.Chrome(chrome_options=options, executable_path=r"C:\Users\a2abv65\Desktop\chromedriver.exe") # location of chrome driver
driver = webdriver.PhantomJS() # if you want to use chrome, replace Firefox() with Chrome()
driver.get("https://quotes.wsj.com/index/XX/BUXX") # load the web page
WebDriverWait(driver, 50).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="quote_dateTime"]')))
#time.sleep(5)

 # waits till the element with the specific id appears
datetime = driver.find_element_by_xpath('//*[@id="quote_dateTime"]').text
quote_val = driver.find_element_by_xpath('//*[@id="quote_val"]').text
quote_change = driver.find_element_by_xpath('//*[@id="quote_change"]').text
quote_change_per = driver.find_element_by_xpath('//*[@id="quote_changePer"]').text
prior_close = driver.find_element_by_xpath('//*[@id="compare_divId"]/div[3]/ul[1]').text
#open_price = driver.find_element_by_xpath('//*[@id="compare_divId"]/div[3]/ul[1]/li[1]/span[2]').text
driver.quit()



d = {'current': [quote_val],'change in dollar' : [quote_change], 'change in percentage': [quote_change_per], 'date_updated':[datetime],'previous close': [prior_close]}
df = pd.DataFrame(data=d)
#df = pd.DataFrame(data=None, columns=df1.columns,index=df1.index)
#df = pd.DataFrame(data=None,columns=['current','change in dollar','change in percentage','previous close','date_updated'])
#df.insert(quote_val,quote_change,quote_change_per,prior_close,datetime)

'''
#datetime
//*[@id="quote_dateTime"]
#current price
//*[@id="quote_val"]
#dollar change
//*[@id="quote_change"]
#percentage dollar change
//*[@id="quote_changePer"]
#prior close
//*[@id="compare_divId"]/div[3]/ul[1]/li[2]/span[2]/text()
#open
//*[@id="compare_divId"]/div[3]/ul[1]/li[1]/span[2]
'''


###########################
cc = pd.DataFrame()     
for s in [' Close','Change']:
    for i in ['WTI']:
        cc = cc.append( pd.DataFrame(
             { 'corrcoef':np.corrcoef(LS[s],LS[i])[0,1] }, index=[s]))
        
features_full = LS[[' Close','Change']]#[columns]
features = features_full[:'2017']
labels= LS[['WTI']][:'2017']
features_test = features_full['2018':]
labels_test = LS[['WTI']]['2018':]

best_params = {'learning_rate': 0.05, 'loss': 'huber', 'max_depth': 5, 'min_samples_split': 20, 'n_estimators': 100}
best_params['n_estimators']=5000

y= pd.Series(labels.as_matrix().squeeze())
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
import pickle
model = LinearRegression()
model.fit(X_train, y_train)
pickle.dump(model, open("WTI_linear.dat", "wb"))


loaded_model = pickle.load(open('WTI_linear.dat', "rb"))
# make predictions for test data

y_pred = loaded_model.predict(features_test)

predictions = pd.DataFrame(y_pred,index=features_test.index)

combo = predictions.join(features_test)
combo.sort_index(inplace=True)
combo.columns = ['forecasted','close','change']
mae = mean_absolute_error(combo['close'], combo['forecasted'])
mse = mean_squared_error(combo['close'], combo['forecasted'])
med = median_absolute_error(combo['close'], combo['forecasted'])
mape = np.mean(np.abs((combo['close'] -  combo['forecasted']) / combo['close'] )) * 100
medape = np.median(np.abs((combo['close'] -  combo['forecasted']) / combo['close'] )) * 100
rmse = np.sqrt(mse)
max_miss = max(abs(combo['close'] - combo['forecasted']))
r2 = r2_score(combo['close'], combo['forecasted'])
print(mae,mse,med,mape,medape,rmse,max_miss,r2)
import matplotlib.pyplot as plt
plt.plot(combo)         
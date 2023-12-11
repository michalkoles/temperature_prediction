import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

csv_path1 = 'C:/csvs/2022.csv'
csv_path2 = 'C:/csvs/2021.csv'
csv_path3 = 'C:/csvs/2020.csv'
csv_path4 = 'C:/csvs/2019.csv'
csv_path5 = 'C:/csvs/2018.csv'
csv_path6 = 'C:/csvs/2017.csv'
csv_path7 = 'C:/csvs/2016-2.csv'
csv_path8 = 'C:/csvs/2015.csv'
csv_path9 = 'C:/csvs/2014.csv'

df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)
df3 = pd.read_csv(csv_path3)
df4 = pd.read_csv(csv_path4)
df5 = pd.read_csv(csv_path5)
df6 = pd.read_csv(csv_path6)
df7 = pd.read_csv(csv_path7)
df8 = pd.read_csv(csv_path8)
df9 = pd.read_csv(csv_path9)

# sloučení dat
df = pd.concat([df9, df8, df7, df6,df5, df4, df3, df2, df1], ignore_index=True)

#filtrování dat
df.index = pd.to_datetime(df['DATE'], format='%Y-%m-%dT%H:%M:%S')
df = df[df['REPORT_TYPE'].str.startswith('FM-12', na=False)]

#převedení teploty na celsie
def metar_temperature_to_celsius(metar_temperature):    
    sign = 1 if metar_temperature[0] == '+' else -1
    temperature_celsius = sign * (float(metar_temperature[1:5])/10)    
    return temperature_celsius

df['TMP'] = df['TMP'].apply(metar_temperature_to_celsius)

temp = df['TMP']

#příprava dat do správného formátu
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    #začátek dat od 2018
    indexStart = df.index.get_loc('2018-01-01 00:00:00')
    for i in range(indexStart,len(df_as_np)-window_size):
        #5 po sobě jdoucích hodin
        row = [[a] for a in df_as_np[i:i+window_size]]
        #6 hodnota která se bude předpovídat
        label = df_as_np[i+window_size]

        date = df.index[i+window_size]
        date_year_max = date.year - 4

        counter = 0

        #4 hodnoty z předchozích let
        while date.year > date_year_max:
            #případ kdy má únor 29 dní
            if date.month == 2 and date.day == 29:
               date = date.replace(day=date.day-1) 

            #přídání do řezezce hodnot
            date = date.replace(year=date.year-1)
            if date in df.index :
                row.append([df[date]])
                counter += 1


        #nahrazení chybějích hodin
        for i in range(counter,4):
            row.append(row[4])
            counter +=1


        X.append(row)
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(temp, WINDOW_SIZE)

X_train, y_train = X[:30000], y[:30000]
X_val, y_val = X[30000:35000], y[30000:35000]
X_test, y_test = X[35000:], y[35000:]

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError 

#model
modelWeather = Sequential()
modelWeather.add(InputLayer((9,1)))
modelWeather.add(LSTM(64))
modelWeather.add(Dense(8,'relu'))
modelWeather.add(Dense(1,'linear'))

#trénování
cp = ModelCheckpoint('modelWeather/', save_best_only=True)
modelWeather.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

modelWeather.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

#načtení modelu
from keras.models import load_model
modelWeather = load_model('modelWeather/')

#testování
train_predictions = modelWeather.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})

print(train_results)

#plt.plot(train_results['Train Predictions'][:100])
#plt.plot(train_results['Actuals'][:100])

val_predictions = modelWeather.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})

##plt.plot(val_results['Val Predictions'][:100])
##plt.plot(val_results['Actuals'][:100])

test_predictions = modelWeather.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})

plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])

plt.show()
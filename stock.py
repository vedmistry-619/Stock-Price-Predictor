import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

dataset = pd.read_csv('NSE-Tata-Global-Beverages-Limited.csv')
'''def conv_dates_series(df, col, old_date_format, new_date_format):
    df[col] = pd.to_datetime(df[col], format=old_date_format).dt.strftime(new_date_format)
    return(df)
conv_dates_series(dataset, "Date", '%m/%d/%Y', '%Y-%m-%d')'''

dataset["Date"]=pd.to_datetime(dataset.Date,format="%Y-%m-%d")
dataset.index=dataset['Date']
plt.figure(figsize=(16,8))
plt.plot(dataset["Close"],label='Close Price history')

data = dataset.sort_index(ascending=True, axis=0)
new_ds = pd.DataFrame(index=range(0,len(dataset)), columns=['Date','Close'])

for i in range(0,len(data)):
    new_ds["Date"][i]=data["Date"][i]
    new_ds["Close"][i]=data["Close"][i]
    
scaler=MinMaxScaler(feature_range=(0,1))
final_dataset=new_ds.values

train_data=final_dataset[0:950,:]
valid_data=final_dataset[950:,:]
new_ds.index=new_ds.Date
new_ds.drop("Date",axis=1,inplace=True)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(new_ds)
x_train,y_train=[],[]
for i in range(60,len(train_data)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
inputs_data=new_ds[len(new_ds)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train,y_train,epochs=2,batch_size=1,verbose=1)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

lstm_model.save("saved_model.h5")

train_data=new_ds[:950]
valid_data=new_ds[950:]
valid_data['Predictions']=predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])
from pandas.io import excel
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import datetime
from datetime import datetime, date, time
import matplotlib.pyplot as plt
#matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10 
#from keras.models import Sequential
import tensorflow as tf
from tf.keras import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler



st.set_page_config(page_title='Survey data 2017-2020.') 
st.header('Cement Plant data analysis')
st.subheader('Rollerpress data')




####--- LOAD DATAFRAME
excel_file ='Down_timeproject_data.xlsx'
sheet_name='Combined downtime'


df = pd.read_excel(excel_file,
                    sheet_name=sheet_name,
                    usecols='B:K',
                    header=2)



st.dataframe(df)







 #-----STREAMLIT SELECTION
responsible=df['Responsible'].unique().tolist()
downtime=df['Downtime (hrs)'].unique().tolist()



year_selection = st.slider('Downtime (Hrs)',
                   min_value=min(downtime),
                   max_value=max(downtime),
                   value=(min(downtime),max(downtime)))


responsible_selection =st.multiselect('Responsible:',
                                responsible,
                                default=responsible)

#---- Filter dataframe based on selection
mask=(df['Downtime (hrs)'].between(*year_selection))&(df['Responsible'].isin(responsible_selection))
number_of_result=df[mask].shape[0]
st.markdown(f'*Available Results: {number_of_result}*')





#--- Group
df_group=df[mask].groupby(by='DATE').count()[['Downtime (hrs)']]
df_group=df_group.rename(columns={'date':'Downtime (hrs)'})
df_group=df_group.reset_index()


#----Plot bar chart
bar_chart = px.bar(df_group,
                    x='DATE',
                    y='Downtime (hrs)',
                    text='Downtime (hrs)',
                    color_discrete_sequence=['#8b0000']*len(df_group),
                    template='plotly_white')

st.plotly_chart(bar_chart)



df1 = pd.DataFrame({

  'date': df['DATE'],
  'Downtime (hrs)': df['Downtime (hrs)']
})

df1

st.line_chart(df1.rename(columns={'date':'index'}).set_index('index'))


# column the charts
col1, col2 = st.beta_columns(2)

##filter single category and add downtime
col1 =df.groupby(["Responsible"])["Downtime (hrs)"].count()
col1




#----PLOT PIE CHART ON DOWNTIME
random_x = col1.values
names = col1.index
  
col2 = px.pie(values=random_x, names=names)
#col2.header("PIE CHART ON DOWNTIME")
st.plotly_chart(col2)





# column the charts
col3, col4 = st.beta_columns(2)

##filter single category and add downtime
col3 =df.groupby(["D.T Category"])["Downtime (hrs)"].count()
col3




#----PLOT PIE CHART ON category
random_x = col3.values
names = col3.index
  
col4 = px.pie(values=random_x, names=names)
st.plotly_chart(col4)





# column the charts
col5, col6 = st.beta_columns(2)

##filter single category and add downtime
col5 =df.groupby(["EQUIPMENT"])["Downtime (hrs)"].count()
col5




#----PLOT PIE CHART ON category
random_x = col5.values
names = col5.index
  
col6 = px.pie(values=random_x, names=names)
st.plotly_chart(col6)



#predict for future incidents


#get rid of the other columns
df2 = df[['DATE', 'Downtime (hrs)']]
df2

#Now, we can convert the “Downtime (hrs)” data type to float
df2 = df2.astype({"Downtime (hrs)": float})
df2["DATE"] = pd.to_datetime(df2.DATE, format="%m/%d/%Y")
df2.dtypes

#Index Column
df.index = df['DATE']

#plot  the data on a graph
plt.plot(df['Downtime (hrs)'],label='Downtime tred')


#Data preparation
df2 = df2.sort_index(ascending=True,axis=0)
data = pd.DataFrame(index=range(0,len(df2)),columns=['DATE','Downtime (hrs)'])



for i in range(0,len(data)):
    data["DATE"][i]=df2['DATE'][i]
    data["Downtime (hrs)"][i]=df2["Downtime (hrs)"][i]
    
data



#max min scaler
scaler=MinMaxScaler(feature_range=(0,1))
data.index=data.DATE
data.drop('DATE',axis=1,inplace=True)
final_data = data.values
train_data=final_data[0:24,:]
valid_data=final_data[24:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_data)
x_train_data,y_train_data=[],[]
for i in range(6,len(train_data)):
    x_train_data.append(scaled_data[i-6:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data
y_train_data

#Long Short-Term Memory Model
lstm_model=Sequential()
lstm_model.add(LSTM(units=1,return_sequences=True,input_shape=(np.shape(x_train_data)[0],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
model_data=data[len(data)-len(valid_data)-20:].values
model_data=model_data.reshape(-1,1)
model_data=scaler.transform(model_data)


#This step covers the preparation of the train data and the test data
#lstm_model.compile(loss='mean_squared_error',optimizer='adam')
#lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)
#X_test=[]
#for i in range(2,model_data.shape[0]):
#    X_test.append(model_data[i-2:i,0])
#X_test=np.array(X_test)
#X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))



#In this step, we are running the model using the test data we defined in the previous step
#predicted_stock_price=lstm_model.predict(X_test)
#predicted_stock_price=scaler.inverse_transform(predicted_stock_price)


#Prediction results

#train_data=data[:200]
#valid_data=data[200:]
#valid_data['Predictions']=predicted_stock_price
#plt.plot(train_data["Close"])
#plt.plot(valid_data[['Close',"Predictions"]])



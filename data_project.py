from pandas.io import excel
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
import datetime
from datetime import datetime, date, time
import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping



st.set_page_config(page_title='Survey data 2017-2020.') 
st.header('Cement Plant Down time data analysis')
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
st.write("Downtime count as per depertment and hours taken")


bar_chart = px.bar(df_group,
                    x='DATE',
                    y='Downtime (hrs)',
                    text='Downtime (hrs)',
                    color_discrete_sequence=['#8b0000']*len(df_group),
                    template='plotly_white')

st.plotly_chart(bar_chart)


st.write("Downtime hrs line chart")
df1 = pd.DataFrame({

  'date': df['DATE'],
  'Downtime (hrs)': df['Downtime (hrs)']
})

df1

st.line_chart(df1.rename(columns={'date':'index'}).set_index('index'))

st.write("Downtime as per Depertment")
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



st.write("Downtime as per Incident Category")

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




st.write("Downtime as per Equipment")
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
#df2 = df2.sort_index(ascending=True,axis=0)
#data = pd.DataFrame(index=range(0,len(df2)),columns=['DATE','Downtime (hrs)'])
#


#for i in range(0,len(data)):
#    data["DATE"][i]=df2['DATE'][i]
 #   data["Downtime (hrs)"][i]=df2["Downtime (hrs)"][i]
    


training_set =  df.iloc[:155, 6:7].values
test_set =  df.iloc[155:, 6:7].values


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(5,155):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#(150, 5, 1)

model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 5, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 5, return_sequences = True))
model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 5, return_sequences = True))
model.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 5))
model.add(Dropout(0.2))# Adding the output layer
model.add(Dense(units = 1))# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 10, batch_size = 3)

# Getting the predicted stock price of 2017
dataset_train =  df.iloc[:159, 6:7]
dataset_test =  df.iloc[:159, 6:7]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 5:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(5,164):
    X_test.append(inputs[i-5:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (120, 40, 1)


predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

 #Visualising the results
#plt.plot( df['DATE'],dataset_test.values, color = 'red', label = 'Real DOWN TIME PERIODS')
#plt.plot( df['DATE'],predicted_stock_price, color = 'blue', label = 'Predicted Down times Hrs')
#plt.xticks(np.arange(0,120,5))
#plt.title('Down time period Prediction')
#plt.xlabel('Time')
#plt.ylabel('Down time period')
#plt.legend()
#plt.show()
st.write("INCIDENT PREDICTION.")
#with st.echo(code_location='below'):
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(
   df["DATE"],
   dataset_test.values,color = 'red', label = 'Real DOWN TIME PERIODS'
  )

ax.scatter(
    df['DATE'],
    predicted_stock_price, color = 'blue', label = 'Predicted Down times Hrs'
    )
ax.set_xlabel("DATE")
ax.set_ylabel("Down time period")

st.write(fig)

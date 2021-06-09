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
from tensorflow import keras
from keras import Sequential
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




training_set =  df.iloc[:, 6:7].values
test_set =  df.iloc[:, 6:7].values


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(5,159 ):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#(740, 60, 1)

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
dataset_train =  df.iloc[:, 6:7]
dataset_test =  df.iloc[:, 6:7]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 6:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(5,159):
    X_test.append(inputs[i-5:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (459, 60, 1)


predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot( df['DATE'],dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
plt.plot( df['DATE'],predicted_stock_price, color = 'blue', label = 'Predicted TESLA Stock Price')
plt.xticks(np.arange(0,159,5))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()

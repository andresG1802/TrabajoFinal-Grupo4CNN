# Importar la librería
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from copy import copy
#plots time series statistics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
#This cell plots the first measurement -> last measurement for each sensor
from datetime import timedelta, datetime
# Leer el dataset
aqi_data = pd.read_csv(r"C:\Users\andre\OneDrive\Documentos\TF-Paralela\US_AQI.csv", index_col=0)


print("Sample of AQI Dataset:")
print(aqi_data.head())


# Parte 2

#dataset summary statistics
aqi_data.describe()

# Parte 3

#plots a map of sensor locations
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12,12))
m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
parallels = np.arange(0.,81,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])
# convert to map projection coords.
# Note that lon,lat can be scalars, lists or numpy arrays.
xpt,ypt = m(aqi_data.lng.values,aqi_data.lat.values)
m.scatter(xpt,ypt, color='black', s=0.1)  # plot a blue dot there
plt.gcf().text(0.2, 0.85, 'Location of Air Quality Measurements', 
               {'size': 25, 'weight': 'bold'})
plt.show()

#Parte 5

# Sets pathces for different aqi levels
from matplotlib.patches import Patch
from copy import copy

GOOD = plt.Rectangle((-1_000,0), 17_500, 50, fc='green',ec="green", alpha=0.25)
MODERATE = plt.Rectangle((-1_000,50), 17_500, 50, fc='yellow',ec="yellow", alpha=0.25)
UNHEALTHY_FOR_SENSITIVE = plt.Rectangle((-1_000,100), 17_500, 50,
                                                  fc='orange',ec="orange", alpha=0.25)
UNHEALTHY = plt.Rectangle((-1_000,150), 17_500, 50, fc='red',ec="red", alpha=0.25)
VERY_UNHEALTHY = plt.Rectangle((-1_000,200), 17_500, 100, fc='purple',ec="purple", alpha=0.25)
HAZARDOUS = plt.Rectangle((-1_000,300), 17_500, 200, fc='maroon',ec="maroon", alpha=0.25)

# Parte 6
#plots time series of US average AQI
aqi_all_usa = aqi_data.groupby('Date').AQI.mean().loc[:'2022-01-01']

aqi_all_usa.plot(figsize=(12,4), color='black', alpha=0.75)
plt.title('USA Average AQI 1980-2022', {'size': 25, 'weight': 'bold'})

# adds color patches 
plt.gca().add_patch(copy(GOOD))
plt.gca().add_patch(copy(MODERATE))
plt.gca().add_patch(copy(UNHEALTHY_FOR_SENSITIVE))
plt.gca().add_patch(copy(UNHEALTHY))
plt.gca().add_patch(copy(VERY_UNHEALTHY))
plt.gca().add_patch(copy(HAZARDOUS))

plt.xlabel('')
plt.ylabel('AQI', {'size': 14, 'weight': 'bold'})

plt.ylim(0, 500)

#Sets up legend
legend_elements = [
    Patch(facecolor='green', edgecolor='black', alpha=0.5, label='Good'),
    Patch(facecolor='yellow', edgecolor='black', alpha=0.5, label='Moderate'),
    Patch(facecolor='orange', edgecolor='black', alpha=0.5, label='Unhealthy for Sensitive Groups'),
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Unhealthy'),
    Patch(facecolor='purple', edgecolor='black', alpha=0.5, label='Very Unhealthy'),
    Patch(facecolor='maroon', edgecolor='black', alpha=0.5, label='Hazardous'),
                  ]


legend = plt.legend(handles=legend_elements)
legend.set_title("AQI Category", prop = {'size':12, 'weight': 'bold'})

#displays the plot
plt.show()

#Parte 7
 
#plots time series statistics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

#decomposes time series seasonality, trend, and residual
seasonal_decompose(aqi_all_usa.values, model='multiplicative', period=365).plot()
plt.gcf().set_figheight(8)
plt.gcf().set_figwidth(12)
plt.show()

#plots partial autocorrelation function
plot_pacf(aqi_all_usa.values, lags=365)
plt.gcf().set_figheight(4)
plt.gcf().set_figwidth(12)
plt.show()

#plots autocorrelation function
plot_acf(aqi_all_usa.values, lags=365)
plt.gcf().set_figheight(4)
plt.gcf().set_figwidth(12)
plt.show()

# Parte 8 
#function to plot aqi given an area
def make_aqi_by_area_plot(df, area, clip_dates=True):
    if clip_dates:
        data = df.unstack().loc[area, :"2022-01-01"]
    else:
        data = df.unstack().loc[area]
        
    data.plot(figsize=(12,4), color='black', alpha=0.75)
    
    
    plt.title(f'{area} Average AQI 1980-2022', {'size': 25, 'weight': 'bold'})

    # adds color patches 
    plt.gca().add_patch(copy(GOOD))
    plt.gca().add_patch(copy(MODERATE))
    plt.gca().add_patch(copy(UNHEALTHY_FOR_SENSITIVE))
    plt.gca().add_patch(copy(UNHEALTHY))
    plt.gca().add_patch(copy(VERY_UNHEALTHY))
    plt.gca().add_patch(copy(HAZARDOUS))

    plt.xlabel('')
    plt.ylabel('AQI', {'size': 14, 'weight': 'bold'})
    
    plt.ylim(0, 500)
    
    #sets up legend
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.5, label='Good'),
        Patch(facecolor='yellow', edgecolor='black', alpha=0.5, label='Moderate'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.5, label='Unhealthy for Sensitive Groups'),
        Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Unhealthy'),
        Patch(facecolor='purple', edgecolor='black', alpha=0.5, label='Very Unhealthy'),
        Patch(facecolor='maroon', edgecolor='black', alpha=0.5, label='Hazardous'),
                      ]

    legend = plt.legend(handles=legend_elements)
    legend.set_title("AQI Category", prop = {'size':12, 'weight': 'bold'})
    
    #display plot
    plt.show()

# Parte 9

#Makes dataframe of US State AQI averages
aqi_by_state = aqi_data.groupby(['state_id', 'Date']).AQI.mean()

#plots AQI averages for each state listed
for state in ['CA', 'TX', 'FL', 'NY']:
    make_aqi_by_area_plot(aqi_by_state, state)

# Parte 10 

#Makes dataframe of AQI by cbsa code
aqi_by_cbsa = aqi_data.groupby(['CBSA Code', 'Date']).AQI.mean()

#plots AQI for each cbsa code listed
for cbsa in [36540, 40140, 19380]:
    make_aqi_by_area_plot(aqi_by_cbsa, cbsa, clip_dates=False)

# Parte 11

#This cell plots the first measurement -> last measurement for each sensor
from datetime import timedelta, datetime

#group data by cbsa and get the first + last measurement
cbsa_grp = aqi_data.groupby('CBSA Code').agg({'Date': ['min', 'max', 'count']})

mins = pd.to_datetime(cbsa_grp[('Date',   'min')])
maxs = pd.to_datetime(cbsa_grp[('Date',   'max')])
min_date = mins.min()

a = mins - min_date
b = maxs - min_date
a = a.dt.days
b = b.dt.days


cbsa_grp['Start'] = a
cbsa_grp['End'] = b
                    
cbsa_grp = cbsa_grp.sort_values(['Start', 'End'], ascending=True)


plt.figure(figsize=(20,10))
plt.hlines(range(671), xmin=cbsa_grp['Start'],
            xmax=cbsa_grp['End'], color='black', 
            alpha=0.5)
plt.plot(cbsa_grp['Start'], range(671), "o", label='First Measurement')
plt.plot(cbsa_grp['End'], range(671), "o", label='Last Measurement')

plt.yticks([671, int(671*0.75), int(671*0.5), int(671*0.25), 0],
           ['100%', '75%', '50%', '25%', '0%'])
plt.xticks(range(0,16_000, 2_000), 
           [str(min_date + timedelta(days=days))[:10] for days in range(0,16_000, 2_000)])
plt.ylabel('Percentage of Sites', {'size': 14, 'weight': 'bold'})

plt.legend(fontsize=20)
plt.grid(axis='both', lw=2, linestyle='-.')

plt.title('Gaps In Measurement For Each CBSA', {'size': 25, 'weight': 'bold'})

plt.show()

# Parte 12 

#This cell plots the number of missing measurements for each sensor
plt.figure(figsize=(8, 8))

plt.barh(range(671),
         ((cbsa_grp['End'] - cbsa_grp['Start']) - cbsa_grp[('Date', 'count')]+1).sort_values().values,
        color='purple', alpha=0.7)

plt.yticks([671, int(671*0.75), int(671*0.5), int(671*0.25), 0],
           ['100%', '75%', '50%', '25%', '0%'])
plt.ylabel('Percentage of Sites', {'size': 14, 'weight': 'bold'})
plt.xlabel('Number of Days with Missing Values', {'size': 14, 'weight': 'bold'})

plt.grid(axis='both', lw=1, linestyle='-.')
plt.title('Number Of Missing Measurements For Each CBSA', {'size': 25, 'weight': 'bold'})
plt.show()

# Parte 13 

#plots the split of training, validation, and training data

aqi_all_usa.plot(figsize=(12,4), color='black', alpha=0.75)
plt.title('Train/Validation/Test Set Split', {'size': 25, 'weight': 'bold'})

# adds color patches 
historic = plt.Rectangle((-1_000,0), 12_688, 500, fc='red',ec="red", alpha=0.25)
train = plt.Rectangle((11_688,0), 3_287, 500, fc='green',ec="green", alpha=0.25)
validation = plt.Rectangle((14_975,0), 188, 500, fc='orange',ec="orange", alpha=0.25)
test = plt.Rectangle((15_163,0), 17_500, 500, fc='blue',ec="blue", alpha=0.25)


plt.gca().add_patch(historic)
plt.gca().add_patch(train)
plt.gca().add_patch(validation)
plt.gca().add_patch(test)


plt.xlabel('')
plt.ylabel('AQI', {'size': 14, 'weight': 'bold'})

plt.ylim(0, 500)
plt.xlim(0, 15_350)

#Sets up legend

legend_elements = [
    Patch(facecolor='red', edgecolor='black', alpha=0.5, label='Historic Data (Not Used)'),
    Patch(facecolor='green', edgecolor='black', alpha=0.5, label='Training Set'),
    Patch(facecolor='orange', edgecolor='black', alpha=0.5, label='Validation Set'),
    Patch(facecolor='blue', edgecolor='black', alpha=0.5, label='Test Set')
                  ]


legend = plt.legend(handles=legend_elements)
legend.set_title("Data Set", prop = {'size':12, 'weight': 'bold'})

#displays the plot
plt.show()

# Parte 14 

# 1. Extract Timeseries Windows
#train_meta_data = []
train_windows = []
#test_meta_data = []
test_windows = []
#val_meta_data = []
val_windows = []


for i in range(cbsa_grp.shape[0]): #for each senseor
    cbsa = cbsa_grp.index[i]
    
    temp_df = aqi_data[aqi_data['CBSA Code'] == cbsa]
    temp_df.index = pd.to_datetime(temp_df['Date'])
    
    #lat = temp_df['lat'].values[0]
    #lng = temp_df['lng'].values[0]
    #population = temp_df['population'].values[0]
    #density = temp_df['density'].values[0]
    
    temp_df = temp_df.loc[:, 'AQI'] # get aqi values

    start_date = temp_df.index.min() # first measurement
    end_date = temp_df.index.max() # last measurement

    
    #gets missing dates and fills it with np.NaN
    new_index = pd.date_range(start_date, end_date, freq='D')
    filler_aqi = [np.NaN for i in new_index]

    filler = pd.DataFrame({'Fill_AQI': filler_aqi}, index=new_index)
    temp = filler.join(temp_df)
    
    aqi_vals = temp.AQI.values
    
    ix = 0
    
    #for each 37 day window
    while ix < aqi_vals.shape[0] - 37: 
        window = aqi_vals[ix: ix+37] 
        
        #if there are no missing values
        if (np.isnan(window).sum() == 0) and ((start_date + timedelta(days=ix)).year > 2011): 

            curr_time = start_date + timedelta(days=ix)
            #doy = curr_time.timetuple().tm_yday
            #dow = curr_time.weekday()
            #sin_doy = np.sin(2*np.pi*doy/366)
            #cos_doy = np.cos(2*np.pi*doy/366)
            #sin_dow = np.sin(2*np.pi*dow/6)
            #cos_dow = np.cos(2*np.pi*dow/6)
            #meta = [sin_doy, cos_doy, sin_dow, cos_dow, lat, lng, population, density]
            
            # 2021 data -> test + val set 
            if curr_time.year == 2021:
                if curr_time.month > 6:
                    #test_meta_data.append(meta)
                    # will limit max aqi value to 500 if over 500
                    test_windows.append([w if w <= 500 else 500 for w in window])
                else:
                    #val_meta_data.append(meta)
                    val_windows.append([w if w <= 500 else 500 for w in window])
                    
            #2012-2020  -> train set
            else:
                #train_meta_data.append(meta)
                train_windows.append([w if w <= 500 else 500 for w in window])
        ix += 1

print('NUMBER OF TRAINING DATA SAMPLES:', len(train_windows))
print('NUMBER OF VALIDATION DATA SAMPLES:', len(val_windows))
print('NUMBER OF TESTING DATA SAMPLES:', len(test_windows))

# Parte 15 

#from sklearn.preprocessing import StandardScaler

#train_meta_data = np.array(train_meta_data)
#test_meta_data = np.array(test_meta_data)
#val_meta_data = np.array(val_meta_data)

#ss_meta = StandardScaler()
#train_meta_data = ss_meta.fit_transform(train_meta_data)
#test_meta_data = ss_meta.transform(test_meta_data)
#val_meta_data = ss_meta.transform(val_meta_data)

# 2. Scale Data
aqi_mean = np.array(train_windows).mean() #mean of training set
aqi_std = np.array(train_windows).std() #standard deviation of training set

#scales data
train_windows = (np.array(train_windows) - aqi_mean) / aqi_std
test_windows = (np.array(test_windows) - aqi_mean) / aqi_std
val_windows = (np.array(val_windows) - aqi_mean) / aqi_std


# 3. Split Into X and y
train_X = train_windows[:, :-7]
train_X = train_X.reshape(train_X.shape[0], 30, 1)
train_y = train_windows[:, -7:]

test_X = test_windows[:, :-7]
test_X = test_X.reshape(test_X.shape[0], 30, 1)
test_y = test_windows[:, -7:]

val_X = val_windows[:, :-7]
val_X = val_X.reshape(val_X.shape[0], 30, 1)
val_y = val_windows[:, -7:]


#display a sample of data
print("SAMPLE OF PREPROCCESED DATA SET")
print("======================================")
print("X:", train_X[0])
print("y:", train_y[0])

# Parte 18 

#Defines Model Architecture
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Bidirectional(LSTM(150, dropout=0.3, return_sequences=True, input_shape=(30,1))),
    Bidirectional(LSTM(150, dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(150, dropout=0.3)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7)
])

model.compile(
    loss='mae',
    optimizer=Adam(learning_rate=1e-5),
    metrics=['mse']
)

# Parte 19

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25
)


history = model.fit(
    x=train_X, 
    y=train_y,
    validation_data=(val_X, val_y),
    epochs=1000,
    batch_size=4_096,
    callbacks=[early_stop],
    verbose=0
)

print("MODEL SUMMARY")
print("==========================")

model.summary()
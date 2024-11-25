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


# Muestra aleatoria de 200,000 filas
aqi_data = aqi_data.sample(n=200000, random_state=42)

# Imprime un vistazo de la muestra
print("Sample of AQI Dataset:")
print(aqi_data.head())

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


# A partir de aqui utiliza el 100 por ciento del procesador
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
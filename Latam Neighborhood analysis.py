#!/usr/bin/env python
# coding: utf-8

# <a href="https://pt.wikipedia.org/wiki/Lima"><img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Lima_-_Per%C3%BA.jpg/840px-Lima_-_Per%C3%BA.jpg" width = 800> </a>

# # Analysing the Development of Businesses in Lima districts

# In[ ]:





# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

from bs4 import BeautifulSoup
import requests

print('Libraries imported.')


# In[2]:


lima_data = "https://es.wikipedia.org/wiki/Anexo:Distritos_de_Lima#Distritos_de_Lima"
lima_dataextracted = requests.get(lima_data).text


# In[3]:


page=BeautifulSoup(lima_dataextracted,"html.parser")
page


# In[4]:


response_obj = requests.get(lima_data).text
soup = BeautifulSoup(response_obj,'lxml')
Districts_lima = soup.find('table',{'class':'wikitable sortable'})
Districts_lima


# In[5]:


totals=Districts_lima.find_all('tr')
nrows=len(totals)
nrows


# In[6]:


header=totals[0].text.split()
header


# In[7]:


totals[2].text


# In[8]:


totals[2].text.split('\n')


# In[9]:


District=totals[1].text.split('\n')[1]
District


# In[10]:


records =[]
n=1
while n < nrows :
    Distritos=totals[n].text.split('\n')[1]
    drop1=totals[n].text.split('\n')[2]
    Ubigeo=totals[n].text.split('\n')[3]
    drop2=totals[n].text.split('\n')[4]
    Área=totals[n].text.split('\n')[5]
    drop3=totals[n].text.split('\n')[6]
    Población=totals[n].text.split('\n')[7]
    drop4=totals[n].text.split('\n')[9]
    Densidad=totals[n].text.split('\n')[8]
    drop5=totals[n].text.split('\n')[10]
    Fundado=totals[n].text.split('\n')[11]
    en=totals[n].text.split('\n')[12]
  
    records.append((Distritos,drop1,Ubigeo,drop2,Área,drop3,Población,drop4,Densidad,drop5,Fundado,en))
    n=n+1

df=pd.DataFrame(records, columns=["Distritos","drop1","Ubigeo","drop2","Área","drop3","Población","Densidad","drop4","drop5","Fundado","en"])
df.head(10)


# In[11]:


df.drop(['drop1','Ubigeo','drop2','drop3','drop4','drop5','Fundado','en'], axis=1, inplace= True)
df


# In[12]:


df.describe


# In[74]:


df.dtypes
df.columns


# In[79]:


df['Área']=df['Área'].astype(float)
df['Población']=df['Población'].astype(float)
df.dtypes


# In[80]:


get_ipython().system('conda install -c anaconda xlrd --yes')


# In[81]:


# use the inline backend to generate the plots within the browser
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0


# In[101]:


df_gra = df.loc[:,'Distritos':'Área']
df_gra.head()


# In[102]:


df_gra.set_index('Distritos',inplace= True)
df_gra.head()


# In[108]:


# step 2: plot data
df_gra.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Districts') # add to x-label to the plot
plt.ylabel('Área') # add y-label to the plot
plt.title('Lima districts Areas') # add title to the plot

plt.show()


# In[110]:


df_gra1 = df.loc[:,'Distritos':'Población']
df_gra1.drop(columns= 'Área', inplace= True)
df_gra1.head()


# In[111]:


df_gra1.set_index('Distritos',inplace= True)
df_gra1.head()


# In[112]:


# step 2: plot data
df_gra1.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Districts') # add to x-label to the plot
plt.ylabel('Poblation') # add y-label to the plot
plt.title('Lima districts Poblation') # add title to the plot

plt.show()


# </div>
#  
# <hr>

# In[20]:


from urllib.request import urlopen as uReq
import requests
import lxml
import pandas as pd
from pandas import DataFrame
import numpy as np


# In[21]:


data =  [['Lima01', 'Lima', -14.046071,-75.704294], ['Lima02', 'Ancon', -11.696553,-77.111654],['Lima03', 'Ate Vitarte', -14.046071,-75.704294],
        ['Lima04', 'Barranco', -12.143959,-77.0202268],['Lima05', 'Breña', -12.059700,-77.050118],['Lima06', 'Carabayllo', -11.794993,-76.989292],
        ['Lima07', 'Comas', -11.932861,-77.040674],['Lima08', 'Chaclacayo', -11.992479,-76.776176],['Lima09', 'Chorrillos', -12.192349,-77.008962],
        ['Lima10', 'El Agustino', -12.042052,-76.995714],['Lima11', 'Jesús María', -12.078186,-77.046411],['Lima12', 'La Molina', -12.090176,-76.922337],
        ['Lima13', 'La Victoria', -12.073357,-77.016417],['Lima14', 'Lince', -12.086567,-77.036647],['Lima15', 'Lurigancho', -11.948832,-76.762701],
        ['Lima16', 'Lurin', -12.238049,-76.783862],['Lima17', 'Magdalena del Mar', -12.491734,-75.911147],['Lima18', 'Miraflores', -12.121498,-77.025906],
        ['Lima19', 'Pachacamac', -12.251096,-76.906592],['Lima20', 'Pucusana', -12.482091,-76.797452],['Lima21', 'Pueblo Libre', -12.076638,-77.076638],
        ['Lima22', 'Puente Piedra', -11.876827,-77.074482],['Lima23', 'Punta Negra', -12.365557,-76.795190],['Lima24', 'Punta Hermosa', -12.332678,-76.825698],
        ['Lima25', 'Rimac', -12.020304,-77.035462],['Lima26', 'San Bartolo', -12.387071,-76.777945],['Lima27', 'San Isidro', -12.097902,-77.035366],
        ['Lima28', 'Independencia', -11.989307,-77.047330],['Lima29', 'San Juan de Miraflores', -12.159910,-76.969140],['Lima30', 'San Luis', -12.072355,-76.995890],
        ['Lima31', 'San Martín de Porres', -11.986759,-77.097655],['Lima32', 'San Miguel', -12.078655,-77.095283],['Lima33', 'Santiago de Surco', -12.125104,-76.981919],
        ['Lima34', 'Surquillo', -12.114197,-77.010474],['Lima35', 'Villa  Maria del triunfo', -12.176643,-76.918967],['Lima36', 'San Juan de Lurigancho', -11.948832,-76.762701],
        ['Lima37', 'Santa Maria del Mar', -12.401402,-76.775465],['Lima38', 'Santa Rosa', -12.035851,-77.086616],['Lima39', 'Los Olivos', -11.965985,-77.073071],
        ['Lima40', 'Cieneguilla', -12.073166,-76.777071],['Lima41', 'San Borja', -12.096451,-76.995689],['Lima42', 'Villa El Salvador', -12.213503,-76.937026],
        ['Lima43', 'Santa Anita', -12.223382,-76.847707]]
newdf = pd.DataFrame(data, columns = ['PostCode', 'District', 'Latitude', 'Longitude'])
newdf


# In[22]:


get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library


# In[23]:


get_ipython().system('conda install -c conda-forge geopy --yes')
#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

# import k-means from clustering stage
from sklearn.cluster import KMeans


# In[24]:


#mapping Lima
latitude = -12.046374
longitude= -77.042793
# create map of Lima using latitude and longitude values above:
map_lima = folium.Map(location=[latitude, longitude], zoom_start=12)


# In[25]:


for lat, lng, label in zip(newdf['Latitude'],newdf['Longitude'], newdf['District']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=25,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.3,
        parse_html=False).add_to(map_lima)  
    
map_lima


# In[26]:


CLIENT_ID = 'HYA5PUMBLRHBCJYOIANTFG1QYS3W4DZIYYYFVSVTJNNJQ0XY' # my Foursquare ID
CLIENT_SECRET = '0UYOK4AJCKHACBATY2ABEH2LZ4J1U4LWV5OO0MOCX3R5I14N' # my Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[32]:


LIMIT = 500 # limit of number of venues returned by Foursquare API
radius = 3000 # define radius


# In[33]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['District', 
                  'District Latitude', 
                  'District Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category' ]
    
    return(nearby_venues)


# In[34]:


lima_venues = getNearbyVenues(names=newdf['District'],
                                   latitudes=newdf['Latitude'],
                                   longitudes=newdf['Longitude']
                                    )


# In[35]:


print(lima_venues.shape)
lima_venues.head(15)


# In[36]:


lima_venues.groupby('District').count()


# In[37]:


print('The number of unique categories is {}.'.format(len(lima_venues['Venue Category'].unique())))


# In[38]:



# one hot encoding
lima_onehot = pd.get_dummies(lima_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
lima_onehot['District'] = lima_venues['District'] 

# move district column to the first column
cols=list(lima_onehot.columns.values)
cols.pop(cols.index('District'))
lima_onehot=lima_onehot[['District']+cols]

# rename Neighborhood for Districts so that future merge works
lima_onehot.rename(columns = {'District': 'District'}, inplace = True)
lima_onehot.head(15)


# In[39]:


lima_onehot.shape


# In[40]:


lima_grouped = lima_onehot.groupby('District').mean().reset_index()
lima_grouped


# In[41]:



lima_grouped.shape


# In[42]:


num_top_venues = 5

for hood in lima_grouped['District']:
    print("----"+hood+"----")
    temp = lima_grouped[lima_grouped['District'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[43]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[44]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['District']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
districts_venues_sorted = pd.DataFrame(columns=columns)
districts_venues_sorted['District'] = lima_grouped['District']

for ind in np.arange(lima_grouped.shape[0]):
    districts_venues_sorted.iloc[ind, 1:] = return_most_common_venues(lima_grouped.iloc[ind, :], num_top_venues)

districts_venues_sorted


# In[51]:


from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs

print('Libraries imported.')


# In[56]:


from sklearn.preprocessing import StandardScaler

X = df.values[:,2:3]
X = np.nan_to_num(X)
cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset


# In[64]:


num_clusters = 5

k_means = KMeans(init="k-means++", n_clusters=num_clusters, n_init=12)
k_means.fit(cluster_dataset)
labels = k_means.labels_
lima_merged['Cluster Labels'] = labels
lima_merged = lima_merged.join(districts_venues_sorted.set_index('District'), on='District')
print(labels)


# In[65]:


lima_merged.head()


# In[66]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(num_clusters)
ys = [i+x+(i*x)**2 for i in range(num_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(lima_merged['Latitude'], lima_merged['Longitude'], lima_merged['District'], lima_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[67]:


lima_merged.loc[lima_merged['Cluster Labels'] == 0, lima_merged.columns[[0] + list(range(5, lima_merged.shape[1]))]]


# In[68]:


lima_merged.loc[lima_merged['Cluster Labels'] == 1, lima_merged.columns[[0] + list(range(5, lima_merged.shape[1]))]]


# In[69]:


lima_merged.loc[lima_merged['Cluster Labels'] == 2, lima_merged.columns[[0] + list(range(5, lima_merged.shape[1]))]]


# In[70]:


lima_merged.loc[lima_merged['Cluster Labels'] == 3, lima_merged.columns[[0] + list(range(5, lima_merged.shape[1]))]]


# In[71]:


lima_merged.loc[lima_merged['Cluster Labels'] == 4, lima_merged.columns[[0] + list(range(5, lima_merged.shape[1]))]]


# ## Results

# <a href="http://www.cementosinka.com.pe/blog/lima-una-ciudad-cementos-inka/"><img src = "http://www.cementosinka.com.pe/blog/wp-content/uploads/2016/12/Lima-una-ciudad-para-Cementos-Inka.png" width = 400> </a>

# In[ ]:





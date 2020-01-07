#!/usr/bin/env python
# coding: utf-8

# ## Note: I have made single notebook for all three question please scroll down for all review all questions

# In[1]:


import numpy as np 

import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim 

import requests 
from pandas.io.json import json_normalize 

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# ### Question 1
# ### Use the BeautifulSoup package or any other way you are comfortable with to transform the data in the table on the Wikipedia page into the above pandas dataframe

# In[5]:


website_url = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text


# In[4]:


from bs4 import BeautifulSoup
soup = BeautifulSoup(website_url,'lxml')
print(soup.prettify())


# #### As we can see  the tabular data is availabe in table and belongs to class="wikitable sortable"So let's extract only table

# In[8]:


My_table = soup.find('table',{'class':'wikitable sortable'})
My_table


# In[9]:


print(My_table.tr.text)


# In[10]:


headers="Postcode,Borough,Neighbourhood"


# #### Geting all values in tr and seperating each td within by ","

# In[12]:


table1=""
for tr in My_table.find_all('tr'):
    row1=""
    for tds in tr.find_all('td'):
        row1=row1+","+tds.text
    table1=table1+row1[1:]
print(table1)


# #### writing  data into csv file
# 

# In[13]:


file=open("toronto.csv","wb")

file.write(bytes(table1,encoding="ascii",errors="ignore"))


# #### Converting into dataframe

# In[16]:


df = pd.read_csv('toronto.csv',header=None)
df.columns=["Postalcode","Borough","Neighbourhood"]
df.head()


# #### Only processing the cells that have an assigned borough. Ignoring the cells with a borough that is Not assigned. Droping row where borough is "Not assigned"¶ 

# In[19]:


indexNames = df[ df['Borough'] =='Not assigned'].index
df.drop(indexNames , inplace=True)
df.head()


# #### If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough¶ 

# In[20]:


df.loc[df['Neighbourhood'] =='Not assigned' , 'Neighbourhood'] = df['Borough']
df.head()


# #### rows with the same postalcode will combined into one row with the neighborhoods separated with a comma

# In[22]:


result = df.groupby(['Postalcode','Borough'], sort=False).agg( ', '.join)
df_new=result.reset_index()
df_new.head(20)


# In[23]:


df_new.shape


# ## Question 2
# ### Use the Geocoder package or the csv file to create dataframe with longitude and latitude values
# 
# #### We will be using a csv file : http://cocl.us/Geospatial_data

# In[35]:



address="C:/Users/HP/Desktop/Geospatial_Coordinates.csv"


# In[36]:


df_lon_lat = pd.read_csv(address)
df_lon_lat.head()


# In[37]:


df_lon_lat.columns=['Postalcode','Latitude','Longitude']


# In[38]:


df_lon_lat.head()


# In[39]:


Toronto_df = pd.merge(df_new,
                 df_lon_lat[['Postalcode','Latitude', 'Longitude']],
                 on='Postalcode')
Toronto_df


# In[40]:


Toronto_df.shape


# ## Question 3
# ### Explore and cluster the neighborhoods in Toronto
# 
# #### Use geopy library to get the latitude and longitude values of Toronto.
# 

# In[42]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="Toronto")
location = geolocator.geocode(address)
latitude_toronto = location.latitude
longitude_toronto = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude_toronto, longitude_toronto))


# In[43]:


map_toronto = folium.Map(location=[latitude_toronto, longitude_toronto], zoom_start=10)

# add markers to map
for lat, lng, borough, Neighbourhood in zip(Toronto_df['Latitude'], Toronto_df['Longitude'], Toronto_df['Borough'], Toronto_df['Neighbourhood']):
    label = '{}, {}'.format(Neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# ### Define Foursquare Credentials and Version

# In[44]:


CLIENT_ID = 'GUYZYKE5JZINZVRK1A10BAFI33FXPBTTAUCG22I2CHSFYEQO' # your Foursquare ID
CLIENT_SECRET = '04RI121NVRR44VH5DYEU23QNC4RGHTM5H1A2LN0LDYPG5Z11' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[45]:


radius=500
LIMIT=100


# In[46]:


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
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[49]:


toronto_venues = getNearbyVenues(names=Toronto_df['Neighbourhood'],
                                   latitudes=Toronto_df['Latitude'],
                                   longitudes=Toronto_df['Longitude']
                                  )


# In[50]:


toronto_venues.head()


# In[51]:


toronto_venues.shape


# #### Let's check how many venues were returned for each neighborhood

# In[56]:


toronto_venues.groupby('Neighbourhood').count()


# ## Analysing Each Neighborhood

# In[57]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot.head()


# In[58]:


toronto_onehot.shape


# In[59]:


toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped


# #### Let's print each neighborhood along with the top 5 most common venues

# In[61]:


num_top_venues = 5

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ## Let's put that into a pandas dataframe

# #### First, let's write a function to sort the venues in descending order.

# In[63]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# #### Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[65]:


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()


# ## Cluster Neighborhoods
# #### Run k-means to cluster the neighborhood into 5 clusters.

# In[66]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_ 
# to change use .astype()


# #### Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[68]:


# add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster_Labels', kmeans.labels_)

toronto_merged = Toronto_df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!


# #### in some row  there is no data available for some neighbourhood so we drope these raws

# In[70]:


toronto_merged=toronto_merged.dropna()


# In[71]:


toronto_merged['Cluster_Labels'] = toronto_merged.Cluster_Labels.astype(int)


# In[72]:


toronto_merged


# #### Finally, let's visualize the resulting clusters

# In[74]:


# create map
map_clusters = folium.Map(location=[latitude_toronto, longitude_toronto], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster_Labels']):
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


# ## Examine Clusters
# ### Cluster 1

# In[83]:


toronto_merged.loc[toronto_merged['Cluster_Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 2

# In[77]:


toronto_merged.loc[toronto_merged['Cluster_Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 3

# In[78]:


toronto_merged.loc[toronto_merged['Cluster_Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 4

# In[80]:


toronto_merged.loc[toronto_merged['Cluster_Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# ### Cluster 5

# In[82]:


toronto_merged.loc[toronto_merged['Cluster_Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:





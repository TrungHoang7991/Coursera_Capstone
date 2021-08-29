#!/usr/bin/env python
# coding: utf-8

# # SAN JOSE

# ##  **I.Introduction**

# >I am currently a Master student majoring in System Analytic - Industrial Engineering. Next year will be my final year at school and I want to prepare for my next chapter in my life. I have asked myself as well as my friends "Where will be the next place?" , the place that I can easily get familiar with job market, cultural, people and lifestyle. And finally, I found that California is satisfied all of my requirements so that this project I want to analyze and have some insight about this state. 

# > ### Why do I chose California ?
# > * First of all, California is popular for the dream place for Engineer jobs. A lot of tech companies are placing in California so that I can have more choices in the future.
# > * Secondly, the weather in California is similar to my country , it is not as cold as the East Coast of the US so that I can fell comfortable all the time.
# > * Finally, cultural diversity has play an important role for my decison and I think that working in that environment, I can learn not only in work but also in many different aspects of lifestyle. 

# ## **II. Methodology**

#  **1.Data Preparation**
# >> I used several websites to collect data about demographic, neighborhoods for this project such as **[San Jose,California](https://en.wikipedia.org/wiki/San_Jose,_California#cite_note-San_Jose_Weatherbox_NOAA_txt-93)**, **[California Demographic](https://worldpopulationreview.com/states/california-population)** . Moreover, I used **[Google Maps](https://www.google.com/maps)** and **[Foursquare](https://foursquare.com/)** to collect the latitude and longitude for my data. 

#  **2.Data Analysis**
# 
# >>* I will make graphs about California and San Jose demographics to compare and have comments the pros and cons of living in this area. 
# >>* Then I will choose the suitable Neighborhood for myself, breakdown it and have the further analysis. 

# ## **III. Data Analysis**

# In[1]:


import pandas as pd # for data processing
import folium   #for creating maps
import requests  #for retreiving Information from URL
from geopy.geocoders import Nominatim  #converting address to cordinates
from pandas.io.json import json_normalize #converting json to DataFrame 
import numpy as np
from sklearn.cluster import KMeans
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt


# > **1.Comparison of county**

# In[2]:


df2 = pd.read_excel('C:\\Users\\TRUNG HOANG\\Desktop\\San Jose.xlsx','California Population')
df2.columns = ['County','Population','GrowthRate','Density']
df2.sort_values(by='Population',ascending=False,ignore_index=True)
df2.head()


# In[3]:


df2_chart=df2.head(10)
fig = plt.figure(figsize = (10, 8))
county = df2_chart['County'].values

plt.bar(county,df2_chart['Population'])
plt.xlabel("County")
plt.xticks(rotation=70)
plt.ylabel("Population")
plt.title("10 Most population county in California")

plt.show()


# > **2.Comparison of San Jose**

# In[4]:


df1 = pd.read_excel('C:\\Users\\TRUNG HOANG\\Desktop\\San Jose.xlsx','San Jose')
df1_grouped = df1.groupby(['Region','Neighborhood'])
df1_grouped.first()


# There are several regions in San Jose and I decided to take a look at East San Jose because there is a small neighborhood call Little Saigon which is similar to my culture. 

# In[5]:


df = pd.read_excel('C:\\Users\\TRUNG HOANG\\Desktop\\San Jose.xlsx','East SJ')
df


# In[6]:


CLIENT_ID = 'ZJFUOG3KH25RM35JAFRKX4EF115JOREL25SWQV5R3KAALBQ4' # your Foursquare ID
CLIENT_SECRET = '12CIFGXYFN0LEF4DAVGAYYPXYWCM0WIRJQAOPLKJ41LSQE4I' # your Foursquare Secret
ACCESS_TOKEN = 'BZHHTBH2MV5XJTY24L3FJ2UWIK30P2Z2XSK5OPVD0IIGDGWA' # your FourSquare Access Token
VERSION = '20180604'
LIMIT = 50
radius = 500
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[7]:


# reading address from user
address = 'East San Jose, California'
geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
#converting address to coordinates
# reading latitude from location
latitude = location.latitude
# reading longitude from location
longitude = location.longitude
print(latitude,longitude)


# In[8]:


# reading radius from user
#radius = input("Enter the radius for searching : ")
#reading search limit from user
#Limit = input("enter the Limit for Results to display : ")
url  = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)
results = requests.get(url).json()


# In[9]:


items = results['response']['groups'][0]['items']


# > **4.Get some nearby places**

# In[10]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

#flatten JSON, Normalize JSON to Dataframe
dataframe = json_normalize(items) 

# filter columns,consider only required columns
filtered_columns = ['venue.name', 'venue.categories'] + [col for col in dataframe.columns if col.startswith('venue.location.')] + ['venue.id']
nearby = dataframe.loc[:, filtered_columns]

# filter the category for each row
nearby['venue.categories'] = nearby.apply(get_category_type, axis=1)

# clean columns
nearby.columns = [col.split('.')[-1] for col in nearby.columns]

#replce NaN values with Not found in address
nearby['address'] = nearby['address'].fillna("Not found")
nearby.head(10)
nearby.shape


# In[11]:


nearby.head(10)


# In[12]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[13]:


from pandas.io.json import json_normalize
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[14]:


def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
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
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[15]:


SanJose_venues = getNearbyVenues(names=df['Neighborhood'],
                                   latitudes=df['Latitude'],
                                   longitudes=df['Longitude']
                                  )


# In[16]:


print(SanJose_venues.shape)
SanJose_venues.head()


# In[17]:


SanJose_venues.groupby('Neighborhood').count()


# In[18]:


# Analyze each neighborhood
# one hot encoding
SJ_onehot = pd.get_dummies(SanJose_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
SJ_onehot['Neighborhood'] = SanJose_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [SJ_onehot.columns[-1]] + list(SJ_onehot.columns[:-1])
SJ_onehot = SJ_onehot[fixed_columns]

SJ_onehot.head()


# In[19]:


# Group neighbor by mean of category
SJ_grouped = SJ_onehot.groupby('Neighborhood').mean().reset_index()
SJ_grouped


# > **5.Each neighbor with top 5 most common places**

# In[20]:


num_top_venues = 5

for hood in SJ_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = SJ_grouped[SJ_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[21]:


#function to sort the venues in descending order

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[22]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = SJ_grouped['Neighborhood']

for ind in np.arange(SJ_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(SJ_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## Perform Clustering

# In[23]:


#Run k-means to cluster the neighborhood into 5 clusters

# set number of clusters
kclusters = 5

SJ_grouped_clustering = SJ_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(SJ_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[24]:


#create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

SJ_merged = df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
SJ_merged = SJ_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

SJ_merged.head() 


# In[25]:


import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
#visualize the resulting clusters

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(SJ_merged['Latitude'], SJ_merged['Longitude'], SJ_merged['Neighborhood'], SJ_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster)],
        fill=True,
        fill_color=rainbow[int(cluster)],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Examine Clusters

# In[26]:


#Cluster1
SJ_merged.loc[SJ_merged['Cluster Labels'] == 0, SJ_merged.columns[[0] + list(range(4, SJ_merged.shape[1]))]]


# In[27]:


#Cluster2
SJ_merged.loc[SJ_merged['Cluster Labels'] == 1, SJ_merged.columns[[0] + list(range(4, SJ_merged.shape[1]))]]


# In[28]:


#Cluster3
SJ_merged.loc[SJ_merged['Cluster Labels'] == 2, SJ_merged.columns[[0] + list(range(4, SJ_merged.shape[1]))]]


# In[29]:


#Cluster 4
SJ_merged.loc[SJ_merged['Cluster Labels'] == 3, SJ_merged.columns[[0] + list(range(4, SJ_merged.shape[1]))]]


# In[30]:


#Cluster 5
SJ_merged.loc[SJ_merged['Cluster Labels'] == 4, SJ_merged.columns[[0] + list(range(4, SJ_merged.shape[1]))]]


# ## **IV. Conclusion**

# After clustering common venues, we observed that Mexican restaurants are very popular in East San Jose. In my opinion, I think that, Meadowfair and Little Saigon neighborhoods are suitable for me because I love Vietmanese food. Moreover, looking at the common venues of these areas, I can describe these neighborhoods are Asian Town because a lot of Asian food are place there such as : Bubble Tea shop, Asian Restaurant or Coffee shop.

# In[ ]:





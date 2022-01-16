import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import numpy as np
df = pd.read_csv('data/pos.csv')

grp = df.groupby('mmsi')
ship = grp.get_group(210621000)


m = folium.Map(location=[ship['lat'].mean(),ship['lon'].mean()])


# fig = px.scatter_geo(ship,lat='lat',lon='lon', hover_name="mmsi")
# fig.update_layout(title = 'World map', title_x=0.5)
# fig.show()
#plt.plot(ship['lat'],ship['lon'])
#f = plt.gcf()
#print(type(f))
# = plt.plot(ship["lon"], ship["lat"], 'b')
#mplleaflet.display(fig = f)


# import pandas as pd

# geometry = [Point(xy) for xy in zip(ship['lon'], ship['lat'])]
# gdf = GeoDataFrame(ship, geometry = geometry)

# #this is a simple map that goes with geopandas
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# europe = world[world.continent == 'Europe']
# ax = europe.plot()
# ax.set_xlim(ship["lon"].min() - 3, ship["lon"].max()+3)
# ax.set_ylim(ship["lat"].min() -3, ship["lat"].max()+3)

# gdf.plot(ax = ax, marker = 'o', color = 'red', markersize = 15)
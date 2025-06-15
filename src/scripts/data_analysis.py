# import geopandas as gpd
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from matplotlib.colors import LogNorm
#
# df = pd.read_csv("./data/df.csv")
# df['dateIdentified'] = pd.to_datetime(df['dateIdentified'], errors='coerce')
# df['year'] = df['dateIdentified'].dt.year
# country_counts = df[df['year'] >= 2015].groupby('level0Name').size().reset_index(name='count')
#
# shapefile_path = "./countries/ne_110m_admin_0_countries.shp"
# world = gpd.read_file(shapefile_path)
#
# europe = world[world['CONTINENT'] == 'Europe']
#
# name_map = {
#     "Czechia": "Czech Republic",
#     "North Macedonia": "Macedonia",
#     "Slovakia": "Slovak Republic",
#     "United Kingdom": "UK",
#     "Bosnia and Herz.": "Bosnia and Herzegovina"
# }
# country_counts['level0Name'] = country_counts['level0Name'].replace(name_map)
#
# merged = europe.merge(country_counts, left_on='ADMIN', right_on='level0Name', how='left')
# merged['count'] = merged['count'].fillna(100)
#
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# merged.plot(column='count', ax=ax, legend=True, cmap='YlGnBu', edgecolor='black', norm=LogNorm(), legend_kwds={'label': "Number of Images", "shrink": 0.5})
# ax.axis('off')
# ax.set_xlim(-25, 45)   # Longitude
# ax.set_ylim(34, 72)    # Latitude
# plt.tight_layout()
# plt.savefig("europe_country_map.png", dpi=300)
# plt.show()
#
#
# year_counts = df['year'].dropna().astype(int).value_counts().sort_index()
# year_counts = year_counts[(year_counts.index >= 2005) & (year_counts.index <= 2024)]
# plt.figure(figsize=(10, 6))
# year_counts.plot(kind='bar', color='steelblue')
# plt.xlabel("Year")
# plt.ylabel("Number of Images")
# plt.xticks(rotation=45)
# sns.despine(top=True, right=False, left=True, bottom=False)
#
# plt.tight_layout()
# plt.savefig("image_count_per_year.png", dpi=300)
# plt.show()
#

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

# --- Load and process data ---
df = pd.read_csv("./data/df.csv")
df['dateIdentified'] = pd.to_datetime(df['dateIdentified'], errors='coerce')
df['year'] = df['dateIdentified'].dt.year
country_counts = df[df['year'] >= 2015].groupby('level0Name').size().reset_index(name='count')

shapefile_path = "./countries/ne_110m_admin_0_countries.shp"
world = gpd.read_file(shapefile_path)
europe = world[world['CONTINENT'] == 'Europe']

name_map = {
    "Czechia": "Czech Republic",
    "North Macedonia": "Macedonia",
    "Slovakia": "Slovak Republic",
    "United Kingdom": "UK",
    "Bosnia and Herz.": "Bosnia and Herzegovina"
}
country_counts['level0Name'] = country_counts['level0Name'].replace(name_map)

merged = europe.merge(country_counts, left_on='ADMIN', right_on='level0Name', how='left')
merged['count'] = merged['count'].fillna(100)

# --- Setup combined figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.4, 1], 'wspace': 0.2})

# --- Map plot ---
merged.plot(
    column='count',
    ax=ax1,
    legend=True,
    cmap='YlGnBu',
    edgecolor='black',
    norm=LogNorm(),
    legend_kwds={'label': "Number of Images", "shrink": 1 }
)

cbar_ax = ax1.get_figure().axes[-1]  # last axis is usually the colorbar
cbar_ax.tick_params(labelsize=20)    # ticks
cbar_ax.set_ylabel("Number of Images", fontsize=20)
cbar_ax.yaxis.set_label_coords(3.0, 0.5)

ax1.axis('off')
ax1.set_xlim(-14, 45)
ax1.set_ylim(34, 72)

# --- Bar plot ---

year_counts = df['year'].dropna().astype(int).value_counts().sort_index()
year_counts = year_counts[(year_counts.index >= 2005) & (year_counts.index <= 2024)]

cmap = plt.get_cmap('YlGnBu')

# Normalize the data to [0, 1]
norm = Normalize(vmin=year_counts.min(), vmax=year_counts.max())

# Map values to colors
colors = [cmap(norm(value)) for value in year_counts]

bars = year_counts.plot(kind='bar', color=colors, ax=ax2)

# Add black edges to each bar
for bar in bars.patches:
    bar.set_edgecolor('black')
    bar.set_linewidth(0.8)

ax2.set_box_aspect(1)
ax2.set_xlabel("Year", fontsize=20)
ax2.yaxis.set_label_coords(1.25, 0.5)
ax2.yaxis.set_label_position("right")
ax2.set_ylabel("Number of Images", fontsize=20)
ax2.tick_params(axis='x', rotation=45, labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
sns.despine(ax=ax2, top=True, right=False, left=True, bottom=False)

# --- Final layout and save ---
plt.subplots_adjust(
    left=0.03,   # reduce space on the left
    right=0.9,  # reduce space on the right
    top=0.95,    # reduce top padding
    bottom=0.07, # reduce bottom padding
    wspace=0.1   # keep your desired horizontal space between subplots
)

plt.savefig("combined_map_barplot.png", dpi=600)
plt.show()


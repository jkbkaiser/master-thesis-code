import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.4, 1], 'wspace': 0.2})

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

norm = Normalize(vmin=year_counts.min(), vmax=year_counts.max())

colors = [cmap(norm(value)) for value in year_counts]

bars = year_counts.plot(kind='bar', color=colors, ax=ax2)

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

plt.subplots_adjust(
    left=0.03,
    right=0.9,
    top=0.95,
    bottom=0.07,
    wspace=0.1,
)

plt.savefig("combined_map_barplot.png", dpi=600)
plt.show()

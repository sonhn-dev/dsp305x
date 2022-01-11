import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import json


def load_data(filename, train_size=0.9, random_state=13):
    raw_df = pd.read_csv(filename)
    target_col = 'MIS_Status'
    raw_df.dropna(subset=[target_col], inplace=True)

    x_cols = list(raw_df.columns)
    x_cols.remove(target_col)

    x_raw = raw_df[x_cols]
    y_raw = raw_df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x_raw,
                                                        y_raw,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=y_raw)

    return x_train, x_test, y_train, y_test


def plot_geospartial(data, figsize, cmap='OrRd', title=None):
    cmap = cm.get_cmap(cmap)
    vmin = data.min()
    vmax = data.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    with open('data/states.geojson') as json_file:
        json_data = json.load(json_file)

    def plot_state(name, coords):
        polys = []
        x = []
        y = []
        color = cmap(norm(data[name]))
        for c in coords:
            poly_coords = c[0]
            Pol = Polygon(poly_coords,
                          fc=color,
                          fill=True)
            polys.append(Pol)
            x += [x for x, y in poly_coords]
            y += [y for x, y in poly_coords]
        xmin = min(x)
        ymin = min(y)
        xmax = max(x)
        ymax = max(y)
        plt.text((xmin+xmax)/2,
                 (ymin+ymax)/2,
                 name,
                 ha='center',
                 va='center',
                 size=9)
        return polys, xmin, ymin, xmax, ymax

    features = json_data['features']
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    polys = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for feature in features:
        stusps = feature['properties']['STUSPS']
        coords = feature['geometry']['coordinates']
        p, xminstate, yminstate, xmaxstate, ymaxstate = plot_state(stusps, coords)
        polys += p
        xmin.append(xminstate)
        ymin.append(yminstate)
        xmax.append(xmaxstate)
        ymax.append(ymaxstate)

    ax = plt.gca()
    ax.add_collection(PatchCollection(polys, match_original=True))
    plt.xlim(min(xmin)-2, max(xmax)+2)
    plt.ylim(min(ymin)-2, max(ymax)+2)
    plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm),
                 ax=ax,
                 aspect=40)
    plt.xticks([])
    plt.yticks([])
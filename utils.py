from IPython.display import display

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

def get_df_info(df):
    df = pd.concat([df.count(axis=0).T,
                    df.isna().sum(axis=0).T,
                    df.nunique(),
                    df.dtypes],
                   axis='columns')
    df.reset_index(inplace=True)
    df.columns = ['Column', 'Count', 'Null count', 'Num unique', 'Type']
    display(df)

# Function to calculate CHGOFF rate by feature 
def chgoff_rate_by_feature(x_train, y_train, feature):
    tmp = pd.concat([x_train[['LoanNr_ChkDgt', feature]], y_train], axis='columns')
    tmp = tmp.pivot_table(index=feature,
                          columns='MIS_Status',
                          values='LoanNr_ChkDgt',
                          aggfunc='count')
    tmp = tmp['CHGOFF'] / tmp.sum(axis='columns')
    return tmp

def date_value(date_strings, errors='coerce', max_year=2000):
    def transform(date_string):
        result = date_string
        if not pd.isna(date_string):
            year = int('20' + date_string[-2:])
            if year > max_year:
                year -= 100
            result = date_string[:-2] + str(year)
            if len(result)==10:
                result = '0' + result
        return result
    result = date_strings.applymap(transform).apply(
        lambda x: pd.to_datetime(x, format='%d-%b-%Y', errors=errors)
    )
    return result

def amount_value(value_strings):
    def transform(value_string):
        return value_string.replace(',', '')[1:]
    return value_strings.applymap(transform).astype(np.float64)

def plot_hist_and_box_plot(original_amount, amount_transformed, y_train, feature, bins=20):
    fig = plt.figure(figsize=(15, 4))
    fig.add_subplot(1, 3, 1)
    plt.hist(original_amount[feature],
             bins=bins)
    plt.title(f'{feature} (original)')
    plt.xlabel('Amount')
    plt.ylabel('Count')
    fig.add_subplot(1, 3, 2)
    plt.hist(amount_transformed[feature],
             bins=bins)
    plt.title(f'{feature} (transformed)')
    plt.xlabel('Amount')
    fig.add_subplot(1, 3, 3)
    cond = (y_train=='P I F')
    plt.boxplot([amount_transformed.loc[cond, feature],
                 amount_transformed.loc[~cond, feature]],
                labels=['PIF', 'CHGOFF'])
    plt.title(f'CHGOFF rate by {feature} (transformed)')
    plt.xlabel('MIS Status')
    plt.ylabel('Amount')
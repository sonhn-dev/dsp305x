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


def load_dataset(filename, target, train_size=0.9, random_state=13):
    '''
    Load dataset

    Input:
        filename: path to csv
        target: name of target column
        train_size: train set size (0.0 - 1.0)
        random_state: seed value

    Return:
        x_train, x_test, y_train, y_test: Pandas DataFrames
    '''

    raw_df = pd.read_csv(filename)
    raw_df.dropna(subset=[target], inplace=True)

    x_cols = list(raw_df.columns)
    x_cols.remove(target)

    x_raw = raw_df[x_cols]
    y_raw = raw_df[target]

    x_train, x_test, y_train, y_test = train_test_split(x_raw,
                                                        y_raw,
                                                        train_size=train_size,
                                                        random_state=random_state,
                                                        stratify=y_raw)
    return x_train, x_test, y_train, y_test


def plot_geospartial(data, figsize, cmap='OrRd', title=None, vmin=None, vmax=None):
    '''
    Plot geo heat map

    Input:
        data: Pandas Series with State index value
        figsize: Plot size
        cmap: Color map
        title: Plot title
        vmin: cmap min value
        vmax: cmap max value
    '''

    cmap = cm.get_cmap(cmap)
    vmin = vmin or data.min()
    vmax = vmax or data.max()
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
    '''
    Get DataFrame info
    
    Input:
        df: DataFrame

    Return:
        DataFrame with input columns infomation
    '''

    df = pd.concat([df.count(axis=0).T,
                    df.isna().sum(axis=0).T,
                    df.nunique(),
                    df.dtypes],
                   axis='columns')
    df.reset_index(inplace=True)
    df.columns = ['Column', 'Count', 'Null count', 'Num unique', 'Type']
    return df

def target_rate_by_feature(x, y):
    '''
    Get target rate by feature
    
    Input:
        x: Pandas Series, feature
        y: Pandas Series, target

    Return:
        DataFrame
    '''

    result = pd.DataFrame(y).pivot_table(index=y,
                                         columns=x,
                                         values=y.name,
                                         aggfunc='count')

    result = result.fillna(0)
    result = result / result.sum()
    return result.T

def date_value(df, errors='coerce', max_year=2000):
    '''
    Convert string to date

    Input:
        df: Pandas DataFrame
        error: How to deal with error (coerce, raise)
        max_year: Maximum year (to deal with Y2K problem)

    Return:
        Pandas DataFrame
    '''

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
    result = df.applymap(transform).apply(
        lambda x: pd.to_datetime(x, format='%d-%b-%Y', errors=errors)
    )
    return result

def amount_value(df):
    '''
    Convert string to amount

    Input:
        df: Pandas DataFrame

    Return:
        Pandas DataFrame
    '''

    def transform(value_string):
        return value_string.replace(',', '')[1:]
    return df.applymap(transform).astype(np.float64)

def plot_hist_and_boxplot(x, x_transformed, y, target_labels, bins=20):
    '''
    Plot histogram and boxplot of continuos variable

    Input:
        x: Pandas Series, original
        x_transformed: Pandas Series, transform
        y: Pandas Series, target
        target_labels: List of target labels
        bins: Number of histogram bins
    '''

    feature_name = x.name

    fig = plt.figure(figsize=(15, 4))

    fig.add_subplot(1, 3, 1)
    plt.hist(x, bins=bins)
    plt.title(f'{feature_name} (original)')
    plt.xlabel(f'{feature_name}')
    plt.ylabel('Count')

    fig.add_subplot(1, 3, 2)
    plt.hist(x_transformed, bins=bins)
    plt.title(f'{feature_name} (transformed)')
    plt.xlabel(f'{feature_name}')

    fig.add_subplot(1, 3, 3)
    plt.boxplot([x_transformed[y==label] for label in target_labels],
                labels=target_labels)
    plt.title(f'{feature_name} (transformed) by {y.name}')
    plt.xlabel(y.name)
    plt.ylabel('Amount')

def plot_box(x, y, target_labels):
    '''
    Plot boxplot of continuos variable

    Input:
        x: Pandas Series, feature
        y: Pandas Series, target
        target_labels: List of target labels
    '''

    [x[y==label] for label in target_labels]
    plt.boxplot([x[y==label] for label in target_labels],
                labels=target_labels)
    plt.title(f'{x.name} by {y.name}')
    plt.xlabel(f'{y.name}')
    plt.ylabel(f'{x.name}')

def mature_between(disburse_date, term, date_from, date_to):
    '''
    Get maturity dates from disbursement dates and terms

    Input:
        disburse_date: Pandas Series, disbursement date
        term: Pandas Series, term
        date_from: String, mature from this date
        date_to: String, mature to this date
    Return:
        Pandas Series with True for maturity date between date_from, date_to 
    '''

    month_offset = term.map(lambda x: pd.DateOffset(months=x))
    maturity_date = disburse_date + month_offset
    cond = ((maturity_date <= np.datetime64(date_to)) &
            (maturity_date >= np.datetime64(date_from)))
    return cond

def plot_stacked_bars(counts):
    '''
    Plot stacked bars with heights add up to 1

    Input:
        counts: Pandas DataFrame
    '''

    def draw_amt(x, y, bottom):
        for i, val in enumerate(y):
            plt.text(x[i],
                     val/2 + bottom[i],
                     f'{val:.2f}',
                     va='center',
                     ha='center')

    x = np.arange(counts.shape[1])
    width = 0.4
    bottom = np.zeros_like(x, dtype=np.float64)
    for _, y in counts.iterrows():
        plt.bar(x, y, width=width, bottom=bottom)
        draw_amt(x, y, bottom)
        bottom += y

    plt.xlabel(counts.columns.name)
    plt.ylabel(counts.index.name)
    plt.xticks(x, counts.columns)
    plt.title(f'{counts.index.name} by {counts.columns.name}')
    plt.legend(counts.index, loc='upper right')
    plt.margins(x=0.5)

def find_clusters(model, data):
    ''' Plot stacked bars with heights add up to 1

    Input:
        model: Sklearn cluster model
        data: Pandas Series
    '''

    x = data.to_numpy().reshape(-1, 1)
    y = model.fit_predict(x)
    tmp = pd.concat([
        data,
        pd.Series(y, index=data.index, name='Cluster')
    ], axis='columns')
    tmp = tmp.reset_index()
    tmp = tmp.groupby('Cluster', as_index=False).agg({data.name: 'max',
                                                      data.index.name: lambda x: x.to_list()})
    tmp = tmp.sort_values(data.name)
    tmp = tmp.reset_index(drop=True)
    tmp = tmp.drop(columns='Cluster')
    return tmp.to_dict(orient='index')
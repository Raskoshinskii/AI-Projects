import matplotlib.pyplot as plt
import seaborn as sns 

import numpy as np 
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram

sns.set_style("darkgrid")


def plot_series(series, city_zone=230):
    """
    Plots a time series graph 

    ts: pd.DataFrame
        Time Series DataFrame
    city_zone: int
        District Number of New York
    Returns: 
    -------
    None 
    """
    plt.figure(figsize=(16,5))
    plt.grid(True)
    plt.title(f'N_Trips in {city_zone}')
    sns.lineplot(x='tpep_pickup_datetime', y='n_trips', data=series.reset_index(), legend=False);


def plot_decomposition(series, figsize = (12, 9), grid=True):
    """
    Plots Time Series decomposition

    series: pd.Series
        Time Series 
    figsize: tuple
        Figure size
    grid: bool
        Wether to plot a grid 
    Returns:
    -------
    None
    """
    ts_compnts = sm.tsa.seasonal_decompose(series)
    titles = ['Origianl', 'Trend', 'Seasonal', 'Resid']
    fig, ax = plt.subplots(nrows=4, ncols=1,figsize=figsize)
    plt.tight_layout()
    ax[0].set_title('Time Series Decomposition')
    ax[0].plot(series)
    ax[1].plot(ts_compnts.trend)
    ax[2].plot(ts_compnts.seasonal)
    ax[3].plot(ts_compnts.resid)
    for indx, title in enumerate(titles):
        ax[indx].set_ylabel(title)
        ax[indx].grid(grid)


def plot_acf_pacf(series, lags=30, figsize=(12, 7)):
    """
    Plots autocorrelation and partial autocorrelation functions 

    ts: pd.Series
        Time Series
    lags: int 
        Max number of lags to plot
    figsize: tuple 
        Figure size
    Returns:
    -------
    None
    """
    plt.figure(figsize=figsize)
    ax = plt.subplot(211)
    sm.graphics.tsa.plot_acf(series.values, lags=lags, ax=ax)
    plt.grid(True)
    ax = plt.subplot(212)
    sm.graphics.tsa.plot_pacf(series.values, lags=lags, ax=ax)
    plt.grid(True)


def plot_cluster_ts(current_cluster):
    """
    Plots time serieses in a cluster 

    current_cluster: np.array
        Cluster with time serieses
    Returns:
    -------
    None 
    """
    fig, ax = plt.subplots(
        int(np.ceil(current_cluster.shape[0]/4)),4,
        figsize=(45, 3*int(np.ceil(current_cluster.shape[0]/4)))
    )
    fig.autofmt_xdate(rotation=45)
    ax = ax.reshape(-1)
    for indx, series in enumerate(current_cluster):
        ax[indx].plot(series)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show();


def plot_clusters(data, cluster_model, dim_red_algo):
    """
    Plots clusters obtained by clustering model 

    data: pd.DataFrame or np.array
        Time Series Data
    cluster_model: Class
        Clustering algorithm 
    dim_red_algo: Class
        Dimensionality reduction algorithm (e.g. TSNE/PCA/MDS...) 
    Returns:
    -------
    None
    """
    cluster_labels = cluster_model.fit_predict(data)
    centroids = cluster_model.cluster_centers_
    u_labels = np.unique(cluster_labels)
    
    # Centroids 
    plt.figure(figsize=(16, 10))
    plt.scatter(centroids[:, 0] , centroids[:, 1] , s=150, color='r', marker="x")
    
    # Downsize data into 2D
    if data.shape[1] > 2:
        data_2d = dim_red_algo.fit_transform(data)
        for u_label in u_labels:
            cluster_points = data[(cluster_labels == u_label)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=u_label)
    # If features are already downsized
    else:
        for u_label in u_labels:
            cluster_points = data[(cluster_labels == u_label)]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=u_label)     
    plt.title('Clustered Data')
    plt.xlabel("Feature space for the 1st feature")
    plt.ylabel("Feature space for the 2nd feature")
    plt.grid(True)
    plt.legend(title='Cluster Labels');


def plot_dendrogram(data, model, figsize=(16,10), **kwargs):
    """
    Plots a dendogram using HAC 

    data: pd.DataFrame or np.array
        Time Series Data
    model: Class
        Clustering Model 
    figsize: tuple
        Figure size
    Returns:
    -------
    None 
    """
    model.fit(data)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    plt.figure(figsize=figsize, dpi=200)
    dendrogram(linkage_matrix, **kwargs)
    plt.title('Dendogram')
    plt.xlabel('Objects')
    plt.ylabel('Distance')
    plt.grid(False)
    plt.tight_layout();
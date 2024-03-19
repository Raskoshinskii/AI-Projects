
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm

sns.set_style("darkgrid")
warnings.filterwarnings('ignore')


def get_kmeans_results(data, max_clusters=10, metric='euclidean', seed=23):
    """
    Runs KMeans n times (according to max_cluster range)

    data: pd.DataFrame or np.array
        Time Series Data
    max_clusters: int
        Number of different clusters for KMeans algorithm
    metric: str
        Distance metric between the observations
    seed: int
        random seed
    Returns: 
    -------
    None      
    """
    distortions = []
    silhouette = []
    clusters_range = range(1, max_clusters+1)
    
    for K in tqdm(clusters_range):
        kmeans_model = TimeSeriesKMeans(n_clusters=K, metric=metric, n_jobs=-1, max_iter=10, random_state=seed)
        kmeans_model.fit(data)
        distortions.append(kmeans_model.inertia_)
        if K > 1:
            silhouette.append(silhouette_score(data, kmeans_model.labels_))
        
    plt.figure(figsize=(10,4))
    plt.plot(clusters_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method');
    
    plt.figure(figsize=(10,4))
    plt.plot(clusters_range[1:], silhouette, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette')


def predict_cluster_labels(cluster_model, dim_red_algo, data):
    """
    Predicts cluster label for a given time series

    cluster_model: Class
        Clustering algorithm 
    dim_red_algo: Class
        Dimensionality reduction algorithm (e.g. TSNE/PCA/MDS...) 
    data: pd.DataFrame or pd.Series
        Time Series data
    Returns:
    --------
    np.array 
    """
    cluster_data = data.pivot(index='PULocationID', columns='tpep_pickup_datetime', values='n_trips').T
    scaler = StandardScaler() 
    scaled_cluster_data = scaler.fit_transform(cluster_data).T
    
    data_down = dim_red_algo.fit_transform(scaled_cluster_data) 
    
    cluster_labels = cluster_model.predict(data_down)
    return cluster_labels
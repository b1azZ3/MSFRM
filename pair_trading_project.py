# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import pandas_datareader as dr

# Import Model Packages
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

# Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle

import warnings

warnings.filterwarnings('ignore')


# Loading the data
def get_data(file):
    dataset = read_csv(file, index_col=0)
    return dataset


# Exploratory data analysis
def exploratory_analysis(dataset):
    print(dataset.shape)

    # peek at data
    set_option('display.width', 100)
    dataset.head(5)

    # describe data
    set_option('display.precision', 3)
    dataset.describe()


# Data cleaning
def data_cleaning(dataset):
    # Checking for any null values and removing the null values
    print('Null Values =', dataset.isnull().values.any())

    # Getting rid of the columns with more than 30% missing values
    missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
    missing_fractions.head(10)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    dataset.drop(labels=drop_list, axis=1, inplace=True)
    print(dataset.shape)

    # Fill the missing values with the last value available in the dataset.
    dataset = dataset.fillna(method='ffill')
    print(dataset.head(2))
    return dataset


# Data transformation
def data_transformation(dataset):
    """ Producing annual returns and variance """

    # Calculate average annual percentage return and volatilises over a theoretical one-year period
    returns = dataset.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
    data = returns

    # Standardize the dataset features into unit scale
    scaler = StandardScaler().fit(data)
    rescaleddataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    # Summarize and return the transformed data
    rescaleddataset.head(2)
    X = rescaleddataset
    print(X.head(2))
    return data, X


"""
Evaluating Algorithms and Models
We will look at the following models:

1. KMeans
2. Hierarchical Clustering (Agglomerate Clustering)
3. Affinity Propagation 
"""

# K-Means Clustering
from sklearn import metrics


# Finding the optimal number of clusters
def find_optimal_num_of_clustering(X, max_loop=20):
    # Calculating distortion
    distortions = []
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Plot SSE for k
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), distortions)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)
    plt.show()

    # Calculating silhouette score
    silhouette_score = []
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(X)
        silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))

    # Plot silhouette score for k
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), silhouette_score)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)
    plt.show()


# Clustering and Visualisation
def building_kmeans_clustering_model(X, nclust=6):
    # Fit with k-means
    k_means = cluster.KMeans(n_clusters=nclust)
    k_means.fit(X)

    # Extracting labels
    target_labels = k_means.predict(X)

    # Create scatter-plot to visualize the cluster in two-dimensional space
    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_means.labels_, cmap="rainbow", label=X.index)
    ax.set_title('k-Means results')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
    plt.show()

    # Show number of stocks in each cluster
    clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    plt.figure(figsize=(12, 7))
    plt.barh(
        range(len(clustered_series.value_counts())),  # cluster labels, y axis
        clustered_series.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()

    return k_means


# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage, ward


# Building hierarchy graph/dendrogram
def building_hierarchy_graph(X, distance_threshold=13):
    # Calculate linkage
    Z = linkage(X, method='ward')
    print(Z[0])

    # Plot Dendrogram
    plt.figure(figsize=(10, 7))
    plt.title("Stocks Dendrograms")
    dendrogram(Z, labels=X.index)
    plt.show()

    # Choose the threshold cut at 13 yields 4 clusters(test)
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
    print(chosen_clusters['cluster'].unique())


# Clustering and visualization
def building_hierarchical_clustering_model(X, nclust=4):
    hc = AgglomerativeClustering(n_clusters=nclust, affinity='euclidean', linkage='ward')
    clust_labels1 = hc.fit_predict(X)

    # Visualize the result
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels1, cmap="rainbow")
    ax.set_title('Hierarchical Clustering')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.show()

    return hc


# Affinity Propagation
# Building model and visualize the results
def building_affinity_propagation_model(X):
    ap = AffinityPropagation()
    ap.fit(X)
    clust_labels2 = ap.predict(X)

    # Visualize the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels2, cmap="rainbow")
    ax.set_title('Affinity')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.show()

    # Visualize the results
    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    no_clusters = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % no_clusters)

    # Plot exemplars
    X_temp = np.asarray(X)
    # plt.close('all')
    # plt.figure(1)
    # plt.clf()
    fig = plt.figure(figsize=(8, 6))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(no_clusters), colors):
        class_members = labels == k
        cluster_center = X_temp[cluster_centers_indices[k]]
        plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X_temp[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.show()

    # Show number of stocks in each cluster
    clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    # clustered stock with its cluster label
    clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    clustered_series_ap = clustered_series_ap[clustered_series_ap != -1]
    plt.figure(figsize=(12, 7))
    plt.barh(
        range(len(clustered_series_ap.value_counts())),  # cluster labels, y axis
        clustered_series_ap.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()

    return ap


# Cluster Evaluation
from sklearn import metrics


def cluster_evaluation(dataset, X, k_means, hc, ap):
    # show number of stocks in each cluster
    clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    # clustered stock with its cluster label
    clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    clustered_series_ap = clustered_series_ap[clustered_series_ap != -1]

    # Calculating Silhouette Coefficient
    print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
    print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
    print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

    # Visualising the return within a cluster
    # all stock with its cluster label (including -1)
    clustered_series = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
    # clustered stock with its cluster label
    clustered_series_all = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
    clustered_series = clustered_series[clustered_series != -1]

    # Get the number of stocks in each cluster
    counts = clustered_series_ap.value_counts()
    # Let's visualize some clusters
    cluster_vis_list = list(counts[(counts < 25) & (counts > 1)].index)[::-1]
    print(cluster_vis_list)
    CLUSTER_SIZE_LIMIT = 9999
    counts = clustered_series.value_counts()
    ticker_count_reduced = counts[(counts > 1) & (counts <= CLUSTER_SIZE_LIMIT)]
    print("Clusters formed: %d" % len(ticker_count_reduced))
    print("Pairs to evaluate: %d" % (ticker_count_reduced * (ticker_count_reduced - 1)).sum())

    # plot a handful of the smallest clusters
    plt.figure(figsize=(12, 7))
    print(cluster_vis_list[0:min(len(cluster_vis_list), 4)])
    for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
        tickers = list(clustered_series[clustered_series == clust].index)
        means = np.log(dataset.loc[:"2018-02-01", tickers].mean())
        data = np.log(dataset.loc[:"2018-02-01", tickers]).sub(means)
        data.plot(title='Stock Time Series for Cluster %d' % clust)
    # plt.show()

    return ticker_count_reduced, clustered_series


# Pair Selection
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(data, significance=0.05):
    # This function is from https://www.quantopian.com/lectures/introduction-to-pairs-trading
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


def get_pairs(dataset, ticker_count_reduced, clustered_series):
    # create dic for clusters
    cluster_dict = {}
    for i, which_clust in enumerate(ticker_count_reduced.index):
        tickers = clustered_series[clustered_series == which_clust].index
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
            dataset[tickers]
        )
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs

    # Get pairs and unique tickers
    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])
    print("Number of pairs found : %d" % len(pairs))
    print("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))
    print(pairs)
    return pairs


# Pair Visualization
from sklearn.manifold import TSNE
import matplotlib.cm as cm


def visualization_tsne(pairs, X, clustered_series, k_means):
    stocks = list(np.unique(pairs))
    X_df = pd.DataFrame(index=X.index, data=X).T
    in_pairs_series = clustered_series.loc[stocks]
    stocks = list(np.unique(pairs))
    X_pairs = X_df.T.loc[stocks]
    X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)
    k_means = cluster.KMeans(n_clusters=6)
    k_means.fit(X)
    centroids = k_means.cluster_centers_
    plt.figure(1, facecolor='white', figsize=(16, 8))
    plt.clf()
    plt.axis('off')
    for pair in pairs:
        # print(pair[0])
        ticker1 = pair[0]
        loc1 = X_pairs.index.get_loc(pair[0])
        x1, y1 = X_tsne[loc1, :]
        # print(ticker1, loc1)

        ticker2 = pair[1]
        loc2 = X_pairs.index.get_loc(pair[1])
        x2, y2 = X_tsne[loc2, :]

        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=in_pairs_series.values, cmap=cm.Paired)
    plt.title('T-SNE Visualization of Validated Pairs');

    # Zip joins x and y coordinates in pairs
    for x, y, name in zip(X_tsne[:, 0], X_tsne[:, 1], X_pairs.index):
        label = name

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
    plt.show()


# Applying pair trading strategy
import pandas as pd
import QuantGlobal as qg


# XEC: Energy Sector, DXC: Information Technology Sector
def calculate_spread(sector1, sector2):
    sector1_data = qg.download(key="yihao.liang@uconn.edu",
                               strategy='pt_extended',
                               underlying=sector1,
                               from_date='2022-11-29',
                               end_date='2022-11-30')

    sector2_data = qg.download(key="yihao.liang@uconn.edu",
                               strategy='pt_extended',
                               underlying=sector2,
                               from_date='2022-11-29',
                               end_date='2022-11-30')

    # Calculate the total return for each sector using top 5 stocks
    top_five_tickers_1 = sector1_data['Ticker'][:5].tolist()
    top_five_tickers_2 = sector2_data['Ticker'][:5].tolist()
    top_five_stocks_1 = sector1_data[sector1_data['Ticker'].isin(top_five_tickers_1)]
    top_five_stocks_2 = sector2_data[sector2_data['Ticker'].isin(top_five_tickers_2)]

    total_return_sector1 = top_five_stocks_1.groupby('datetime')['Returns'].mean().reset_index()
    total_return_sector2 = top_five_stocks_2.groupby('datetime')['Returns'].mean().reset_index()

    # Calculate the spread
    spread = total_return_sector1['Returns'] - total_return_sector2['Returns']

    # Create a new dataframe with datetime and spread columns
    index_spread = pd.DataFrame({'datetime': total_return_sector1['datetime'], 'spread': spread})

    return index_spread


# Place the order
def place_order(spreads, stock1, stock2):
    # When the index spread crosses to/above this level, we want to put on a trade
    opening_threshold = 0.75

    # We check the most recent value of the spread to see if it is at the threshold
    if abs(spreads['spread'].iloc[-1]) >= opening_threshold:

        if spreads['spread'].iloc[-1] > 0:
            long_order = broker_api.buy(stock2)
            short_order = broker_api.short(stock1)
        else:
            long_order = broker_api.buy(stock1)
            short_order = broker_api.short(stock2)
    else:
        pass


# Close order when spread < 0.25
def close_trade(spreads, stock1, stock2):
    closing_threshold = 0.25

    if spreads['spread'].iloc[-1] <= closing_threshold:
        if spreads['spread'].iloc[-1] > 0:
            long_order = broker_api.buy(stock1)
            short_order = broker_api.short(stock2)
        else:
            long_order = broker_api.buy(stock2)
            short_order = broker_api.short(stock1)
    else:
        pass


if __name__ == '__main__':
    file_ = "D:/5352/pythonProject/SP500Data.csv"
    # Data exploration and transformation
    dataset_ = get_data(file_)
    exploratory_analysis(dataset_)
    data_ = data_cleaning(dataset_)
    _data_, X_ = data_transformation(data_)
    # k_means, hierarchical, affinity_propagation
    find_optimal_num_of_clustering(X_, max_loop=20)
    k_means_ = building_kmeans_clustering_model(X_, nclust=6)
    building_hierarchy_graph(X_, distance_threshold=13)
    hc_ = building_hierarchical_clustering_model(X_, nclust=4)
    ap_ = building_affinity_propagation_model(X_)
    ticker_count_reduced_, clustered_series_ = cluster_evaluation(dataset_, X_, k_means_, hc_, ap_)
    # Find pairs and plot tsne
    pairs_ = get_pairs(data_, ticker_count_reduced_, clustered_series_)
    visualization_tsne(pairs_, X_, clustered_series_, k_means_)
    # Apply trading strategy (given XEC and DXC)
    spreads_ = calculate_spread('energy', 'technology')
    place_order(spreads_, 'XEC', 'DXC')
    close_trade(spreads_, 'XEC', 'DXC')

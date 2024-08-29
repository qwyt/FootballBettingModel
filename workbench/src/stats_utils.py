# TODO: this should be moved to preprocessing etc.
from typing import List, Dict, Union

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler

cutoff_values = [*range(100, 300, 10)]
# cutoff_values = [*range(40, 200, 10)]
# cutoff_values = [50]

EPS_OPTIONS = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # DBSCAN
MIN_SAMPLES = [20, 22, 24, 25, 27, 29, 30, 35, 15, 17]  # DBSCAN

MIN_CLUSTER_ITEMS = 0
USE_HIERARCHICAL = True
USE_DBSCAN = False
#
CLUSTERING_CANDIDATES: Dict[str, List[Union[int, None]]] = {
    "None": [None],
    # "None": [4],
    # "None": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # "PCA": [3, 4, 5, 6, 7, 8, 9, 10],
    # "t-SNE": [2, 3],
}


def find_optimal_clusters(source_df_standardized):
    def cluster_hierarchical(_df, method, cutoff):
        Z = linkage(_df, method=method)
        clusters = fcluster(Z, cutoff, criterion="distance")
        return clusters

    def cluster_db_scan(_df, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(source_df_standardized)
        return clusters

    def get_results(
        df_reduced,
        name,
        clusters_hierarchical,
        method,
        component_method,
        n_components,
        cutoff,
        eps,
        min_samples,
    ):
        unique_values, cluster_counts = np.unique(
            clusters_hierarchical, return_counts=True
        )
        min_count_in_cluster = np.min(cluster_counts)

        pass

        if (
            len(set(clusters_hierarchical)) > 1
            and min_count_in_cluster > MIN_CLUSTER_ITEMS
        ):  # Check to avoid error if only one cluster is found
            score_silhouette = silhouette_score(df_reduced, clusters_hierarchical)
            score_calinski_harabasz = calinski_harabasz_score(
                df_reduced, clusters_hierarchical
            )
            score_davies_bouldin = davies_bouldin_score(
                df_reduced, clusters_hierarchical
            )

            res = {
                "name": name,
                "component_method": component_method,
                "n_components": n_components,
                "method": method,
                "cutoff": cutoff,
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": len(np.unique(clusters_hierarchical)),
                "min_count_in_cluster": min_count_in_cluster,
                "score_silhouette": score_silhouette,
                "score_calinski_harabasz": score_calinski_harabasz,
                "score_davies_bouldin": score_davies_bouldin,
            }
            return res
        return None

    res_df = []

    configurations = [
        (cluster_type, n_component)
        for cluster_type, n_components in CLUSTERING_CANDIDATES.items()
        for n_component in n_components
    ]

    for cluster_type, n_components in configurations:
        if cluster_type == "PCA":
            pca = PCA(n_components=n_components)
            df_reduced = pca.fit_transform(source_df_standardized)
        elif cluster_type == "t-SNE":
            if n_components > 3:
                continue
            tsne = TSNE(n_components=n_components)
            df_reduced = tsne.fit_transform(source_df_standardized)
        else:
            df_reduced = source_df_standardized

        if USE_HIERARCHICAL:
            # for method in ["ward"]:
            for method in ["ward", "single", "average", "weighted", "centroid"]:
                # for method in ['ward', 'single', 'average', 'weighted', 'centroid']:
                Z = linkage(df_reduced, method=method)

                for cutoff in cutoff_values:
                    # Generating cluster labels

                    clusters = cluster_hierarchical(df_reduced, method, cutoff)
                    res = get_results(
                        df_reduced=df_reduced,
                        name="Hierarchical",
                        clusters_hierarchical=clusters,
                        method=method,
                        component_method=cluster_type,
                        n_components=n_components,
                        cutoff=cutoff,
                        eps=None,
                        min_samples=None,
                    )
                    if res:
                        res_df.append(res)

        if USE_DBSCAN:
            for eps in EPS_OPTIONS:
                for min_samples in MIN_SAMPLES:
                    clusters = cluster_db_scan(df_reduced, eps, min_samples)
                    res = get_results(
                        df_reduced=df_reduced,
                        name="DBSCAN",
                        clusters_hierarchical=clusters,
                        method=None,
                        component_method=cluster_type,
                        n_components=n_components,
                        cutoff=None,
                        eps=eps,
                        min_samples=min_samples,
                    )

                    if res:
                        res_df.append(res)

    res_df = pd.DataFrame(res_df)
    if len(res_df) > 0:
        weight_silhouette = 0.4
        weight_calinski_harabasz = 0.4
        weight_davies_bouldin = 0.2

        scaler = MinMaxScaler()

        metrics_df = res_df[
            ["score_silhouette", "score_calinski_harabasz", "score_davies_bouldin"]
        ]

        # Invert Davies-Bouldin scores as lower values are better
        metrics_df["score_davies_bouldin"] = 1 / metrics_df["score_davies_bouldin"]

        metrics_normalized = scaler.fit_transform(metrics_df)
        metrics_normalized_df = pd.DataFrame(
            metrics_normalized, columns=metrics_df.columns
        )

        # Calculate weighted combined score
        res_df["score"] = (
            weight_silhouette * metrics_normalized_df["score_silhouette"]
            + weight_calinski_harabasz
            * metrics_normalized_df["score_calinski_harabasz"]
            + weight_davies_bouldin * metrics_normalized_df["score_davies_bouldin"]
        )

        res_df = res_df.sort_values(by="score", ascending=False)
        return res_df

    return None

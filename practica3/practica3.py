#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
Autor: Adrián Lattes Grassi
"""

import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def kmeans_silhouettes(X, ns):
    silhouettes = []
    clusters = []
    for n_clusters in ns:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
        silhouettes.append(silhouette_score(X, labels))
        clusters.append(kmeans)
    return silhouettes, clusters 

def kmeans_elegir_n_clusters(ns, silhouettes, plot = True):
    silhouettes_max_index = silhouettes.index(max(silhouettes))
    n_clusters = ns[silhouettes_max_index]
    plt.title("Valores medios del coeficiente de Silhouette para distintos números de vecindades", fontweight="bold")
    plt.xlabel("Número de vecindades")
    plt.ylabel("Valor medio del coeficiente de Silhouette")
    bars = plt.bar(ns, silhouettes)
    bars[silhouettes_max_index].set_color('r')
    plt.xticks(ns)
    # plt.xticklabels(range_n)
    return n_clusters, silhouettes_max_index

def plot_clusters_voronoi(X, clusters, n_clusters, ax = None):
    if ax is None:
        _, ax = plt.subplots()
    labels = clusters.labels_
    unique_labels = set(labels)
    colors = [cm.nipy_spectral(i/n_clusters) for i in range(len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)
    lims = (ax.get_xlim(), ax.get_ylim())

    # Diagrama de voronoi construido a partir de los centros de las vecindades
    vor = Voronoi(clusters.cluster_centers_)
    voronoi_plot_2d(vor, show_vertices = False, point_size=0, ax=ax)

    # Hack para que voronoi_plot_2d no cambie la escala de la gráfica.
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])


def main():
    # Datos
    centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=2000, centers=centers, cluster_std=0.4, random_state=0)

    # Posibles valores de vecindades
    ns = range(2,16)

    # Cálculo del coeficiente de silhouette de cada n
    silhouettes, clusters = kmeans_silhouettes(X, ns)

    # Selecciono el valor de n con mayor coeficiente de silhouette
    n_clusters, n_clusters_index = kmeans_elegir_n_clusters(ns, silhouettes)
    clusters = clusters[n_clusters_index]

    # Gráfica de la clasificación, con colores y líneas de separación del diagrama de Voronoi
    plot_clusters_voronoi(X, clusters, n_clusters)

    plt.show()


if __name__ == "__main__":
    main()

sys.exit(1)

plt.subplots(1, 2)




clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = clusters.labels_
unique_labels = set(labels)
centers = clusters.cluster_centers_

vor = Voronoi(centers)
fig = voronoi_plot_2d(vor, show_vertices = False, point_size=0)

colors = [cm.nipy_spectral(i / n_clusters) for i in range(len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)
    p2.plot(fig)


plt.show()

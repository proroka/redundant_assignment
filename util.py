from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import matplotlib.cm
import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import time
import sklearn
import sklearn.datasets


def create_grid_map(grid_size=10, edge_length=10.,
                    minimum_time=10., maximum_time=20.,
                    min_stddev=5., max_stddev=10.,
                    sparse_covariance=True):
  nx_graph = nx.grid_2d_graph(grid_size, grid_size)
  num_edges = 2 * len(nx_graph.edges())

  # Build mean travel time for each edge.
  mean_times = np.random.rand(num_edges) * (maximum_time - minimum_time) + minimum_time
  # Build covariance.
  s = time.time()
  print('Creating covariance matrix for {} edges...'.format(num_edges))
  cov_times = random_covariance_v2(num_edges, sparse_covariance=sparse_covariance)
  stddevs = collections.defaultdict(lambda: np.random.rand() * (max_stddev - min_stddev) + min_stddev)
  for i in range(num_edges):
    cov_times[i, i] *= stddevs[i] * stddevs[i]
    for j in range(i + 1, num_edges):
      cov_times[i, j] *= stddevs[i] * stddevs[j]
      cov_times[j, i] *= stddevs[i] * stddevs[j]
  print('Took {} seconds for {} edges'.format(time.time() - s, num_edges))

  graph = nx.DiGraph(mean=mean_times, covariance=cov_times)
  node_to_index = {}
  for i, (x, y) in enumerate(nx_graph.nodes()):
    graph.add_node(i, x=float(x) * edge_length, y=float(y) * edge_length)
    node_to_index[(x, y)] = i
  for i, (u, v) in enumerate(nx_graph.edges()):
    graph.add_edge(node_to_index[u], node_to_index[v], index=2 * i, mean_time=mean_times[i])
    graph.add_edge(node_to_index[v], node_to_index[u], index=2 * i + 1, mean_time=mean_times[i])
  return graph


def k_shortest_paths(graph, source, target, k):
  return list(itertools.islice(nx.shortest_simple_paths(graph, source, target, weight='mean_time'), k))


# Method to generate reasonably nice covariance matrices.
# See https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor
def random_covariance(size, beta=5.):
  p = np.random.beta(beta, beta, size=(size, size))  # Storing partial correletions.
  p = 2 * p - 1.  # Shift between -1 and 1.

  s = np.identity(size)
  for k in range(size - 1):
    for i in range(k + 1, size):
      cov = p[k, i]
      for l in range(k - 1, -1, -1):
        cov *= np.sqrt((1. - p[l, i] ** 2.) * (1. - p[l, k] ** 2.))
        cov += p[l, i] * p[l, k]
      s[k, i] = cov
      s[i, k] = cov
  # Random permutation.
  permutation = np.random.permutation(size)
  s[:, :] = s[permutation, :]
  s[:, :] = s[:, permutation]
  return s


def random_covariance_v2(size, sparse_covariance=True):
  return sklearn.datasets.make_sparse_spd_matrix(
      dim=size, alpha=0.9 if sparse_covariance else 0., norm_diag=True, smallest_coef=0.1, largest_coef=0.9)


def show_edge_time_covariance(graph, ax=None):
  if ax is None:
    ax = plt.gca()
  p = ax.imshow(graph.graph['covariance'])
  plt.colorbar(p, ax=ax)


def show_average_edge_times(graph, ax=None):
  max_time = 0.
  min_time = float('inf')
  for u, v, data in graph.edges(data=True):
    max_time = max(data['mean_time'], max_time)
    min_time = min(data['mean_time'], min_time)
  # Go through all edges in graph, append with corresponding color from map
  cmap = matplotlib.cm.get_cmap('RdYlGn')
  lines = []
  route_colors = []
  max_x = -float('inf')
  min_x = float('inf')
  max_y = -float('inf')
  min_y = float('inf')
  for u, v, data in graph.edges(data=True):
    # If it has a geometry attribute (ie, a list of line segments)
    if 'geometry' in data:
      xs, ys = data['geometry'].xy
      max_x = max(max(xs), max_x)
      min_x = min(min(xs), min_x)
      max_y = max(max(ys), max_y)
      min_y = min(min(ys), min_y)
      lines.append(list(zip(xs, ys)))
    else:
      x1 = graph.node[u]['x']
      y1 = graph.node[u]['y']
      x2 = graph.node[v]['x']
      y2 = graph.node[v]['y']
      max_x = max(max(x1, x2), max_x)
      min_x = min(min(x1, x2), min_x)
      max_y = max(max(y1, y2), max_y)
      min_y = min(min(y1, y2), min_y)
      line = [(x1, y1), (x2, y2)]
      lines.append(line)
    route_colors.append(cmap((data['mean_time'] - min_time) / (max_time - min_time)))
  if ax is None:
    ax = plt.gca()
  lc = LineCollection(lines, colors=route_colors, linewidths=3, alpha=0.5, zorder=3)
  ax.add_collection(lc)
  margin_x = (max_x - min_x) * .02
  margin_y = (max_y - min_y) * .02
  ax.set_xlim(min_x - margin_x, max_x + margin_y)
  ax.set_ylim(min_y - margin_x, max_y + margin_y)
  xaxis = ax.get_xaxis()
  yaxis = ax.get_yaxis()
  xaxis.get_major_formatter().set_useOffset(False)
  yaxis.get_major_formatter().set_useOffset(False)
  ax.axis('off')
  ax.margins(0)
  ax.tick_params(which='both', direction='in')
  xaxis.set_visible(False)
  yaxis.set_visible(False)
  ax.set_aspect('equal')
  cax = ax.imshow([[min_time, max_time]], vmin=min_time, vmax=max_time, visible=False, cmap=cmap)  # This won't show.
  cbar = plt.colorbar(cax, ax=ax, ticks=[min_time, max_time])
  cbar.ax.set_yticklabels(['%d [s]' % min_time, '%d [s]' % max_time])

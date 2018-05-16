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
import numba as nb
import sklearn
import sklearn.datasets


def create_grid_map(grid_size=10, edge_length=10.,
                    minimum_time=10., maximum_time=20.,
                    min_stddev=5., max_stddev=10.,
                    smallest_correlation=.1,
                    largest_correlation=.9):
  nx_graph = nx.grid_2d_graph(grid_size, grid_size)
  num_edges = len(nx_graph.edges())
  mean_times, cov_times = random_edges(num_edges, minimum_time, maximum_time, min_stddev,
                                       max_stddev, smallest_correlation, largest_correlation)

  graph = nx.Graph(mean=mean_times, covariance=cov_times)
  node_to_index = {}
  for i, (x, y) in enumerate(nx_graph.nodes()):
    graph.add_node(i, x=float(x) * edge_length, y=float(y) * edge_length)
    node_to_index[(x, y)] = i
  for i, (u, v) in enumerate(nx_graph.edges()):
    graph.add_edge(node_to_index[u], node_to_index[v], index=i, mean_time=mean_times[i])
  return graph


def create_random_graph(num_nodes=10,
                        minimum_time=10., maximum_time=20.,
                        min_stddev=5., max_stddev=10.,
                        smallest_correlation=.1,
                        largest_correlation=.9):
  d = np.sqrt(1. / num_nodes * 2)
  n_nodes = 0
  n_tries = 5
  while n_tries and n_nodes < .9 * num_nodes:
    nx_graph = nx.random_geometric_graph(num_nodes, d)
    components = nx.connected_component_subgraphs(nx_graph)
    nx_graph = max(components, key=len)
    n_nodes = len(nx_graph.nodes())
    d *= 1.1
    n_tries -= 1
  if n_nodes < .9 * num_nodes:
    raise ValueError('Not enough connectivity.')

  # Rename nodes.
  node_idx = {}
  for i, idx in enumerate(nx_graph.nodes()):
    node_idx[idx] = i
  nx.relabel_nodes(nx_graph, node_idx, copy=False)

  # Counts trues edges.
  edges = set()
  for u, v in nx_graph.edges():
    if u == v:
      continue
    edges.add(tuple(sorted((u, v))))
  num_edges = len(edges)

  mean_times, cov_times = random_edges(num_edges, minimum_time, maximum_time, min_stddev,
                                       max_stddev, smallest_correlation, largest_correlation)
  graph = nx.Graph(mean=mean_times, covariance=cov_times)
  for i, d in nx_graph.nodes(data=True):
    if 'pos' in d:
      x, y = d['pos']
      graph.add_node(i, x=x, y=y)
    elif 'x' in d:
      graph.add_node(i, x=d['x'], y=d['y'])
    else:
      graph.add_node(i)
  idx = 0
  already_added = set()
  for u, v in nx_graph.edges():
    u, v = sorted((u, v))
    if (u, v) not in edges:
      continue
    if (u, v) in already_added:
      continue
    already_added.add((u, v))
    graph.add_edge(u, v, index=idx, mean_time=mean_times[idx])
    idx += 1
  return graph


def random_edges(num_edges, minimum_time=10., maximum_time=20.,
                 min_stddev=5., max_stddev=10.,
                 smallest_correlation=.1,
                 largest_correlation=.9):
  # Build mean travel time for each edge.
  mean_times = np.random.rand(num_edges) * (maximum_time - minimum_time) + minimum_time
  # Build covariance.
  cov_times = random_covariance(num_edges, smallest_correlation, largest_correlation,
                                positive_correlation_only=False)
  stddevs = collections.defaultdict(lambda: np.random.rand() * (max_stddev - min_stddev) + min_stddev)
  for i in range(num_edges):
    cov_times[i, i] *= stddevs[i] * stddevs[i]
    for j in range(i + 1, num_edges):
      cov_times[i, j] *= stddevs[i] * stddevs[j]
      cov_times[j, i] *= stddevs[i] * stddevs[j]
  return mean_times, cov_times


def k_shortest_paths(graph, source, target, k):
  return list(itertools.islice(nx.shortest_simple_paths(graph, source, target, weight='mean_time'), k))


# Method to generate reasonably nice covariance matrices.
# See https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor
def random_covariance(size,
                      smallest_correlation=.1,
                      largest_correlation=.9,
                      beta=5.,
                      positive_correlation_only=False):
  @nb.jit(nopython=True)
  def _helper(p):
    s = np.zeros((size, size))
    for k in range(size - 1):
      for i in range(k + 1, size):
        cov = p[k, i]
        for l in range(k - 1, -1, -1):
          cov *= np.sqrt((1. - p[l, i] ** 2.) * (1. - p[l, k] ** 2.))
          cov += p[l, i] * p[l, k]
        s[k, i] = cov
        s[i, k] = cov
    return s

  p = np.random.beta(beta, beta, size=(size, size))  # Storing partial correletions.
  if not positive_correlation_only:
    p = 2 * p - 1.  # Shift between -1 and 1.
  s = _helper(p)
  max_value = np.max(np.abs(s))
  min_value = np.min(np.abs(s))
  s = (s - min_value) / (max_value - min_value) * (largest_correlation - smallest_correlation) + smallest_correlation
  np.fill_diagonal(s, 1.)
  # Random permutation.
  permutation = np.random.permutation(size)
  s[:, :] = s[permutation, :]
  s[:, :] = s[:, permutation]
  return s


def random_covariance_v2(size, covariance_sparsity=0.7,
                         smallest_correlation=.1,
                         largest_correlation=.9):
  # Only consider positive correlation.
  return sklearn.datasets.make_sparse_spd_matrix(
      dim=size, alpha=covariance_sparsity, norm_diag=True,
      smallest_coef=smallest_correlation, largest_coef=largest_correlation)


def show_edge_time_covariance(graph, ax=None):
  if ax is None:
    ax = plt.gca()
  p = ax.imshow(graph.graph['covariance'])
  plt.colorbar(p, ax=ax)


def show_average_edge_times(graph, ax=None):
  attrs = set(graph.nodes(data=True)[0].keys())
  if 'pos' in attrs:
    pos_x = {}
    pos_y = {}
    print(graph.nodes(data=True))
    for u, (x, y) in graph.nodes(data=True).iteritems():
      pos_x[u] = x
      pos_y[u] = y
    print(pos_x, pos_y)
    nx.set_node_attributes(graph, 'x', pos_x)
    nx.set_node_attributes(graph, 'y', pos_y)

  if 'x' not in attrs:
    pos = nx.spring_layout(graph)
    pos_x = {}
    pos_y = {}
    for u, (x, y) in pos.iteritems():
      pos_x[u] = x
      pos_y[u] = y
    nx.set_node_attributes(graph, 'x', pos_x)
    nx.set_node_attributes(graph, 'y', pos_y)

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
  points = []
  for u, data in graph.nodes(data=True):
    points.append([data['x'], data['y']])
  points = np.array(points, np.float32)
  if ax is None:
    ax = plt.gca()
  ax.scatter(points[:, 0], points[:, 1], c=(.8, .8, .8), edgecolors='k', zorder=2)

  # Plot hubs.
  points = []
  for i in range(5):
    data = graph.nodes(data=True)[i]
    points.append([data['x'], data['y']])
  points = np.array(points, np.float32)
  ax.scatter(points[:, 0], points[:, 1], c='r', edgecolors='k', zorder=2)

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
  lc = LineCollection(lines, colors=route_colors, linewidths=1.5, alpha=0.5, zorder=1)
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

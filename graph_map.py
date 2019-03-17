from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

import util


class GraphMap(object):

  def __init__(self, size, top_k=3, largest_correlation=.9, closest_k=0):
    # Create a simple networkx graph for now.
    self._graph = util.create_random_graph(size, smallest_correlation=0.,
                                           largest_correlation=largest_correlation)
    self._edge_indices = {}
    for (u, v), idx in nx.get_edge_attributes(self._graph, 'index').items():
      self._edge_indices[(u, v)] = idx
      self._edge_indices[(v, u)] = idx
    # Keep track of top_k paths.
    self._topk = top_k
    self._topk_paths = {}
    self._reuse_edge_samples = None
    self._closest_k = closest_k
    self._closestk_nodes = {}
    self._searcher = NearestNeighborSearcher(self._graph)

  @property
  def num_nodes(self):
    return len(self._graph.nodes())

  @property
  def top_k(self):
    return self._topk

  def topk_paths(self, agent, task):
    if (agent, task) not in self._topk_paths:
      self._topk_paths[(agent, task)] = util.k_shortest_paths(self._graph, agent, task, self._topk)
      paths = self._topk_paths[(agent, task)]
      while len(paths) < self._topk:
        paths.append(paths[0])
    return self._topk_paths[(agent, task)]

  def sample_node(self, node):
    if self._closest_k == 0:
      return node
    if node in self._closestk_nodes:
      nodes = self._closestk_nodes[node]
    else:
      x = self._graph.node[node]['x']
      y = self._graph.node[node]['y']
      nodes = self._searcher.SearchK([x, y], k=self._closest_k + 1)[1]
      self._closestk_nodes[node] = nodes
    return nodes[np.random.randint(len(nodes))]

  def sample_edges(self, num_samples, reuse_samples=False, ignore_correlations=False):
    covariance = self._graph.graph['covariance']
    if ignore_correlations:
      covariance = np.diag(np.diag(covariance))
    if reuse_samples:
      assert self._reuse_edge_samples is not None
      edges = self._reuse_edge_samples
      assert num_samples == edges.shape[0]
    else:
      edges = np.random.multivariate_normal(self._graph.graph['mean'], covariance, size=num_samples)
      edges = np.maximum(edges, 0.)
      self._reuse_edge_samples = edges
    return edges

  def sample(self, agents, tasks, num_samples, ignore_correlations=False, reuse_samples=False,
             node_uncertainty=False, edge_uncertainty=True):
    """Samples a 4D `np.ndarray` of assignment weights.

    In particular, the dimensions of the output are:
    1) agent index
    2) task index
    3) path index
    4) Sample index
    """
    if edge_uncertainty:
      edges = self.sample_edges(num_samples, reuse_samples, ignore_correlations)
    else:
      edges = self.sample_edges(1, reuse_samples, ignore_correlations)
      assert self._topk == 1, 'Set top-k to 1 when not using edge uncertainty.'

    # Compute the top-k shortest paths.
    travel_time = np.zeros((len(agents), len(tasks), self._topk, num_samples), dtype=np.float32)
    for s in range(num_samples if node_uncertainty else 1):
      for i, agent in enumerate(agents):
        if node_uncertainty:
          agent = self.sample_node(agent)  # Alter initial node position.
        else:
          pass
        for j, task in enumerate(tasks):
          paths = self.topk_paths(agent, task)
          assert len(paths) == self._topk
          # Get length of paths.
          for k, path in enumerate(paths):
            for edge in zip(path[:-1], path[1:]):
              idx = self._edge_indices[edge]
              if node_uncertainty:
                travel_time[i, j, k, s] += edges[s if edge_uncertainty else 0, idx]
              else:
                travel_time[i, j, k, :] += edges[:, idx]
    return travel_time

  def sample_path(self, agent, task, path_index, num_samples, ignore_correlations=False, reuse_samples=False):
    assert path_index < self._topk
    edges = self.sample_edges(num_samples, reuse_samples, ignore_correlations)
    paths = self.topk_paths(agent, task)
    assert len(paths) == self._topk
    path = paths[path_index]
    travel_time = np.zeros(num_samples, dtype=np.float32)
    for edge in zip(path[:-1], path[1:]):
      idx = self._edge_indices[edge]
      travel_time += edges[:, idx]
    return travel_time

  def show_covariance(self, ax=None):
    util.show_edge_time_covariance(self._graph, ax)

  def show_mean(self, ax=None, num_hubs=10):
    util.show_average_edge_times(self._graph, ax, num_hubs=num_hubs)

  def show(self, ax=None, num_hubs=10):
    util.show_average_edge_times(self._graph, ax, monochrome=True, num_hubs=num_hubs)

  def show_path(self, agent, task, path_index):
    assert path_index < self._topk
    path = self.topk_paths(agent, task)[path_index]
    util.show_route(self._graph, path)

  def show_node(self, node):
    util.show_node(self._graph, node)


class NearestNeighborSearcher(object):

  def __init__(self, graph):
    points = []
    indices = []
    for k, v in graph.node.items():
      points.append([v['x'], v['y']])
      indices.append(k)
    self.indices = np.array(indices)
    self.kdtree = cKDTree(points, 10)

  def Search(self, xy):
    if isinstance(xy, np.ndarray) and xy.shape == 1:
      single_point = True
      xy = [xy]
    else:
      single_point = False
    distances, indices = self.kdtree.query(xy)
    if single_point:
      return self.indices[indices[0]], distances[0]
    return self.indices[indices], distances

  def SearchRadius(self, xy, dist=1.):
    return self.kdtree.query_ball_point(xy, r=dist)

  def SearchK(self, xy, k=1):
    return self.kdtree.query(xy, k=k)


if __name__ == '__main__':
  import matplotlib.pylab as plt

  graph_size = 200
  top_k = 3
  closest_k = 10
  graph = GraphMap(graph_size, top_k, largest_correlation=.9, closest_k=closest_k)

  plt.figure()
  graph.show_mean()
  plt.show()

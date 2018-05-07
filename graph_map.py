from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import time

import util


class GraphMap(object):

  def __init__(self, size, top_k=3, sparse_covariance=True):
    # Create a simple networkx graph for now.
    self._graph = util.create_grid_map(size, sparse_covariance=sparse_covariance)
    print('Number of nodes: {}, number of edges: {}'.format(
        len(self._graph.nodes()), len(self._graph.edges())))
    self._edge_indices = nx.get_edge_attributes(self._graph, 'index')
    # Keep track of top_k paths.
    self._topk = top_k
    self._topk_paths = {}
    self._reuse_edge_samples = None

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

  def sample(self, agents, tasks, num_samples, ignore_correlations=False, reuse_samples=False):
    """Samples a 4D `np.ndarray` of assignment weights.

    In particular, the dimensions of the output are:
    1) agent index
    2) task index
    3) path index
    4) Sample index
    """
    edges = self.sample_edges(num_samples, reuse_samples, ignore_correlations)

    # Compute the top-k shortest paths.
    travel_time = np.zeros((len(agents), len(tasks), self._topk, num_samples), dtype=np.float32)
    for i, agent in enumerate(agents):
      for j, task in enumerate(tasks):
        paths = self.topk_paths(agent, task)
        assert len(paths) == self._topk
        # Get length of paths.
        for k, path in enumerate(paths):
          for edge in zip(path[:-1], path[1:]):
            idx = self._edge_indices[edge]
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

  def show_mean(self, ax=None):
    util.show_average_edge_times(self._graph, ax)


if __name__ == '__main__':
  graph_size = 12
  num_agents = 20
  num_tasks = 20
  top_k = 3
  num_samples = 100

  graph = GraphMap(graph_size, top_k, sparse_covariance=True)
  graph.sample(np.random.randint(graph.num_nodes, size=num_agents),
               np.random.randint(graph.num_nodes, size=num_tasks),
               num_samples)

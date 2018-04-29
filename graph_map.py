from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pylab as plt
import numpy as np
import time

import util


class Agent(object):

  def __init__(self, position):
    self._position = position

  @property
  def position(self):
    return self._position


class Task(object):

  def __init__(self, position):
    self._position = position

  @property
  def position(self):
    return self._position


class GraphMap(object):

  def __init__(self, size):
    # Create a simple networkx graph for now.
    self._graph = util.create_grid_map(size)

  def sample(self, agents, tasks, k, num_samples):
    """Samples a 4D `np.ndarray` of assignment weights.

    In particular, the dimensions of the output are:
    1) agent index
    2) task index
    3) path index
    4) Sample index
    """
    s = time.time()
    np.random.multivariate_normal(self._graph.graph['mean'], self._graph.graph['covariance'], size=200)
    print('Sampling time: {}'.format(time.time() - s))

  def show_covariance(self, ax=None):
    util.show_edge_time_covariance(self._graph, ax)

  def show_mean(self, ax=None):
    util.show_average_edge_times(self._graph, ax)


if __name__ == '__main__':
  graph = GraphMap(4)
  graph.sample([0] * 10, [0] * 10, k=3, num_samples=10)

  plt.figure()
  graph.show_mean()
  plt.figure()
  graph.show_covariance()
  plt.show()

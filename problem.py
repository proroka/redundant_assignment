from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import scipy.optimize as opt
from scipy.stats.stats import pearsonr


class Aggregation(object):

  def __init__(self, basic, arg, elementwise):
    self._basic = basic
    self._arg = arg
    self._elementwise = elementwise

  def along_axis(self, array, axis):
    return self._basic(array, axis=axis)

  def arg_along_axis(self, array, axis):
    return self._arg(array, axis=axis)

  def elementwise(self, array1, array2):
    return self._elementwise(array1, array2)


class MinimumAggregation(Aggregation):

  def __init__(self):
    super(MinimumAggregation, self).__init__(np.amin, np.argmin, np.minimum)


def _hungarian(graph, agents, tasks, num_samples=10,
               aggregation=MinimumAggregation(), samples=None):
  # Samples.
  if samples is None:
    samples = graph.sample(agents, tasks, num_samples)
  else:
    num_samples = samples.shape[-1]
  # Run Hungarian assignment on the average travel times of the fastest path.
  mean_travel = np.mean(samples, axis=-1)
  best_paths = aggregation.arg_along_axis(mean_travel, axis=-1)
  cost_matrix = aggregation.along_axis(mean_travel, axis=-1)  # Pick the fastest path.
  agent_idx, task_idx = opt.linear_sum_assignment(cost_matrix)
  task_assignments = collections.defaultdict(list)
  for i, j in zip(agent_idx, task_idx):
    task_assignments[j].append((i, best_paths[i, j]))
  return task_assignments


def _greedy_dp(graph, deployment_size, agents, tasks, num_samples=10,
               aggregation=MinimumAggregation(), samples=None):
  num_agents = len(agents)
  num_tasks = len(tasks)
  assert num_tasks <= num_agents, 'Not all tasks can be attended to.'

  # Samples.
  if samples is None:
    samples = graph.sample(agents, tasks, num_samples)
  else:
    num_samples = samples.shape[-1]

  # Run Hungarian assignment on the average travel times of the fastest path.
  task_assignments = _hungarian(graph, agents, tasks, aggregation=aggregation, samples=samples)
  task_state = np.empty((num_tasks, num_samples))
  available_agents = set(range(num_agents))
  for j, assignment in task_assignments.iteritems():
    assert len(assignment) == 1, 'Hungarian issue'
    i, k = assignment[0]
    available_agents.remove(i)
    task_state[j, :] = samples[i, j, k, :]

  # Repeat until we reach the desired deployment size.
  for _ in range(min(num_agents - num_tasks, deployment_size - num_tasks)):
    best_improvement = -float('inf')
    best_assignment = None
    best_state = None

    # Go through each possible remaining assigments.
    for i in available_agents:
      for j in range(num_tasks):
        for k in range(graph.top_k):
          # Compute improvement of assigning agent to task through path k.
          state = aggregation.elementwise(samples[i, j, k, :], task_state[j, :])
          improvement = np.mean(task_state[j, :]) - np.mean(state)
          if improvement > best_improvement:
            best_state = state
            best_assignment = (i, j, k)
            best_improvement = improvement
    i, j, k = best_assignment
    task_state[j, :] = best_state
    task_assignments[j].append((i, k))
    available_agents.remove(i)
  return task_assignments


def _random(graph, deployment_size, agents, tasks, num_samples=10,
            aggregation=MinimumAggregation(), samples=None):
  num_tasks = len(tasks)

  # Samples.
  if samples is None:
    samples = graph.sample(agents, tasks, num_samples)
  else:
    num_samples = samples.shape[-1]

  # Run Hungarian assignment on the average travel times of the fastest path.
  task_assignments = _hungarian(graph, agents, tasks, aggregation=aggregation, samples=samples)
  available_agents = set(range(num_agents))
  for j, assignment in task_assignments.iteritems():
    assert len(assignment) == 1, 'Hungarian issue'
    i, k = assignment[0]
    available_agents.remove(i)

  # Add random assignments.
  nd = min(num_agents - num_tasks, deployment_size - num_tasks)
  for i in np.random.choice(list(available_agents), size=nd, replace=False):
    j = np.random.randint(num_tasks)
    k = np.random.randint(graph.top_k)
    task_assignments[j].append((i, k))
  return task_assignments


def _compute_cost(graph, assignments, agents, tasks, aggregation=MinimumAggregation(),
                  samples=None):
  if samples is None:
    samples = graph.sample(agents, tasks, 1)

  cost = 0.
  num_tasks = len(tasks)
  for j in range(num_tasks):
    assert j in assignments, 'Bad assignments'
    i, k = zip(*assignments[j])
    i = np.array(i, dtype=np.int32)
    k = np.array(k, dtype=np.int32)
    cost += np.mean(aggregation.along_axis(samples[i, j, k, :], axis=0)).item()
  return cost


def _average_coalition_correlation(graph, assignments, agents, tasks):
  corrs = 0.
  total = 0
  graph.sample_edges(100)
  for j, assignments in assignments.iteritems():
    if len(assignments) <= 1:
      continue
    corr = 0.
    n = 0
    for i1, k1 in assignments:
      if agents[i1] == tasks[j]:
        continue
      s1 = graph.sample_path(agents[i1], tasks[j], k1, 100, reuse_samples=True)
      for i2, k2 in assignments:
        if i2 <= i1:
          continue
        if agents[i2] == tasks[j]:
          continue
        s2 = graph.sample_path(agents[i2], tasks[j], k2, 100, reuse_samples=True)
        corr += np.abs(pearsonr(s1, s2)[0])
        n += 1
    if n == 0:
      continue
    corr /= n
    corrs += corr
    total += 1
  if total == 0:
    return float('nan')
  return corrs / total


class Problem(object):

  def __init__(self, graph, agents, tasks, num_samples=10, num_groundtruth_samples=1,
               aggregation=MinimumAggregation()):
    self._graph = graph
    self._agents = agents
    self._tasks = tasks
    self._num_samples = num_samples
    self._num_groundtruth_samples = num_groundtruth_samples
    self._aggregation = aggregation
    self.reset()

  def reset(self):
    self._samples = graph.sample(self._agents, self._tasks, self._num_samples)
    self._gt_sample = graph.sample(self._agents, self._tasks, self._num_groundtruth_samples)

  def hungarian(self, deployment_size=0):
    # Deployment size is ignored.
    self._assignments = _hungarian(self._graph, self._agents, self._tasks,
                                   aggregation=self._aggregation, samples=self._samples)
    return _compute_cost(self._graph, self._assignments, self._agents, self._tasks,
                         aggregation=self._aggregation, samples=self._gt_sample)

  def greedy(self, deployment_size):
    self._assignments = _greedy_dp(self._graph, deployment_size, self._agents, self._tasks,
                                   aggregation=self._aggregation, samples=self._samples)
    return _compute_cost(self._graph, self._assignments, self._agents, self._tasks,
                         aggregation=self._aggregation, samples=self._gt_sample)

  def no_correlation_greedy(self, deployment_size):
    samples = graph.sample(self._agents, self._tasks, self._num_samples, ignore_correlations=True)
    self._assignments = _greedy_dp(self._graph, deployment_size, self._agents, self._tasks,
                                   aggregation=self._aggregation, samples=samples)
    return _compute_cost(self._graph, self._assignments, self._agents, self._tasks,
                         aggregation=self._aggregation, samples=self._gt_sample)

  def random(self, deployment_size):
    self._assignments = _random(self._graph, deployment_size, self._agents, self._tasks,
                                aggregation=self._aggregation, samples=self._samples)
    return _compute_cost(self._graph, self._assignments, self._agents, self._tasks,
                         aggregation=self._aggregation, samples=self._gt_sample)

  def get_correlation(self):
    return _average_coalition_correlation(self._graph, self._assignments, self._agents, self._tasks)


if __name__ == '__main__':
  import graph_map
  import tqdm

  graph_size = 12
  num_agents = 20
  num_tasks = 4
  deployment = 6
  top_k = 3
  num_samples = 1000
  gt_num_samples = 10
  num_loops = 10

  graph = graph_map.GraphMap(graph_size, top_k, sparse_covariance=True)
  agents = np.random.randint(graph.num_nodes, size=num_agents)
  tasks = np.random.randint(graph.num_nodes, size=num_tasks)

  problem = Problem(graph, agents, tasks, num_samples=num_samples,
                    num_groundtruth_samples=gt_num_samples,
                    aggregation=MinimumAggregation())

  # Solve n times.
  costs = {
      'hungarian': ([], []),
      'greedy': ([], []),
      'random': ([], []),
      'no_correlation_greedy': ([], []),
  }
  for _ in tqdm.tqdm(range(num_loops)):
    problem.reset()
    for algorithm, (values, corrs) in costs.iteritems():
      values.append(getattr(problem, algorithm)(deployment))
      corrs.append(problem.get_correlation())

  for algorithm, (values, corrs) in costs.iteritems():
    print('Cost (%s): %g - correlation: %g' % (algorithm, np.mean(values), np.mean(corrs)))

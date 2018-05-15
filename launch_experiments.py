from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from concurrent import futures
import msgpack
import msgpack_numpy
import numpy as np
import os
import tempfile
import tqdm
import threading
import subprocess
import sys

import graph_map
import problem

_NUM_AGENTS = 25
_NUM_HUBS = 5
_NUM_TASKS = 5
_NUM_NODES = 200
_NUM_GRAPHS = 1000
_NUM_SAMPLES = 200
_NUM_SAMPLES_GT = 10
_NUM_THREADS = 24

# Variable.
_CORRELATION_STRENGTH = np.linspace(.1, .9, 9).tolist()
_DEPLOYMENT_SIZE = range(_NUM_TASKS, _NUM_AGENTS + 1, 2)
_TOP_K = [1, 2, 4, 8, 16]

# Fixed.
_BASE_CORRELATION_STRENGTH = .9
_BASE_DEPLOYMENT_SIZE = 20
_BASE_TOP_K = 4


Arguments = collections.namedtuple('Arguments', [
    'deployment_size', 'top_k', 'correlation_strength'])
Arguments.__new__.__defaults__ = (_BASE_DEPLOYMENT_SIZE, _BASE_TOP_K, _BASE_CORRELATION_STRENGTH)

OldArguments = collections.namedtuple('OldArguments', [
    'deployment_size', 'top_k', 'correlation_sparsity', 'correlation_strength'])
OldArguments.__new__.__defaults__ = (_BASE_DEPLOYMENT_SIZE, _BASE_TOP_K, .3, _BASE_CORRELATION_STRENGTH)


def store_results(results, filename):
  with open(filename, 'wb') as fp:
    buf = msgpack.packb(results, use_bin_type=True)
    fp.write(buf)


def read_results(filename):
  with open(filename, 'rb') as fp:
    r = msgpack.unpackb(fp.read(), raw=False)
  os.remove(filename)
  return r


def run_problem(filename, arguments):
  graph = graph_map.GraphMap(_NUM_NODES, arguments.top_k,
                             largest_correlation=arguments.correlation_strength)
  agents = np.random.randint(_NUM_HUBS, size=_NUM_AGENTS)
  tasks = np.random.randint(_NUM_HUBS, graph.num_nodes, size=_NUM_TASKS)
  p = problem.Problem(graph, agents, tasks, num_samples=_NUM_SAMPLES,
                      num_groundtruth_samples=_NUM_SAMPLES_GT,
                      aggregation=problem.MinimumAggregation())

  results = {
      'lower_bound': ([], []),
      'hungarian': ([], []),
      'repeated_hungarian': ([], []),
      'greedy': ([], []),
      'random': ([], []),
      'no_correlation_greedy': ([], []),
  }
  p.reset()
  for algorithm, (costs, correlations) in results.items():
    cost = getattr(p, algorithm)(arguments.deployment_size)
    correlation = p.get_correlation()
    costs.extend(cost)
    correlations.append(correlation)

  store_results(results, filename)


def run_task(filename, arguments):
  args = [sys.executable, __file__, '--output', filename]
  for field in Arguments._fields:
    args.append('--{}'.format(field))
    args.append(str(getattr(arguments, field)))
  return subprocess.call(args)


def done(fn, counter):
  counter.inc()


class AtomicProgressBar(object):
  def __init__(self, total):
    self._value = 0
    self._lock = threading.Lock()
    self._tqdm = tqdm.tqdm(total=total)

  def inc(self):
    with self._lock:
      self._value += 1
      self._tqdm.update(1)

  def close(self):
    self._tqdm.close()


def run(final_filename):
  directory = tempfile.mkdtemp()

  args = set()
  for d in _DEPLOYMENT_SIZE:
    args.add(Arguments(deployment_size=d))
  for top_k in _TOP_K:
    args.add(Arguments(top_k=top_k))
  for correlation_strength in _CORRELATION_STRENGTH:
    args.add(Arguments(correlation_strength=correlation_strength))
  all_args = []
  for arg in sorted(args):
    all_args.extend([arg] * _NUM_GRAPHS)

  threads = []
  executor = futures.ProcessPoolExecutor(max_workers=_NUM_THREADS)
  counter = AtomicProgressBar(len(all_args))
  for i, a in enumerate(all_args):
    filename = os.path.join(directory, 'results_{}.bin'.format(i))
    threads.append((executor.submit(run_task, filename, a), filename, i))
    threads[-1][0].add_done_callback(lambda fn: done(fn, counter))

  all_results = collections.defaultdict(dict)
  for thread, filename, idx in threads:
    if thread.result() != 0:
      raise ValueError('Error while running a process.')
    thread_results = read_results(filename)
    results = all_results[all_args[idx]]
    if not results:
      results.update(thread_results)
      continue
    for algorithm, (costs, correlations) in thread_results.items():
      results[algorithm][0].extend(costs)
      results[algorithm][1].extend(correlations)

  all_results = dict(all_results)  # Remove defaultdict.
  store_results(all_results, final_filename)

  for args, results in all_results.items():
    print('Results for', args)
    baseline_costs = np.array(results['hungarian'][0], np.float32)
    for algorithm, (costs, correlations) in results.items():
      c = np.array(costs, np.float32)
      print('  Cost (%s): %g - correlation: %g' % (algorithm, np.mean(c / baseline_costs), np.mean(correlations)))


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.

  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--output_results', action='store', default=None, help='Where to store results.')

  # Internal arguments.
  parser.add_argument('--output', action='store', default=None)
  defaults = Arguments()
  for field in Arguments._fields:
    v = getattr(defaults, field)
    parser.add_argument('--{}'.format(field), type=type(v), action='store', default=v)
  args = parser.parse_args()

  if args.output:
    run_problem(args.output,
                Arguments(args.deployment_size,
                          args.top_k,
                          args.correlation_strength))
  else:
    assert args.output_results, 'Must specify --output_results'
    run(args.output_results)

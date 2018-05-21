from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import matplotlib.pylab as plt
import msgpack
import msgpack_numpy
import numpy as np
import scipy
import scipy.stats as st
from six.moves import input

import launch_experiments

Data = collections.namedtuple('Data', ['costs', 'correlations'])

_LOWER_BOUND = 'lower_bound'
_UPPER_BOUND = 'hungarian'
_IGNORE = set(['no_correlation_greedy'])

_LINESTYLE = {
    'hungarian': '--',
}
_DEFAULT_LINESTYLE = '-'

_COLORS = {
    'lower_bound': 'black',
    'greedy': 'red',
    'no_correlation_greedy': 'green',
    'hungarian': 'black',
    'repeated_hungarian': 'orange',
    'random': 'blue',
}

_ORDER_AS = [
    'hungarian',
    'random',
    'repeated_hungarian',
    'no_correlation_greedy',
    'greedy',
    'lower_bound',
]


def read_results(filename):
  with open(filename, 'rb') as fp:
    return msgpack.unpackb(fp.read(), raw=False, use_list=False)


def errors(a, use_ci=True):
  if use_ci:
    u, v = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
    return u.item(), v.item()
  m = np.mean(a)
  s = np.std(a)
  return m - s, m + s


def make_nice(bp):
  for box in bp['boxes']:
    box.set(color='k', linewidth=2)
    box.set(facecolor='steelblue')
  for whisker in bp['whiskers']:
    whisker.set(color='k', linewidth=2)
  for cap in bp['caps']:
    cap.set(color='k', linewidth=2)
  for median in bp['medians']:
    median.set(color='k', linewidth=2)
  for flier in bp['fliers']:
    flier.set(marker='o', color='k', alpha=0.5)


def run(filename):
  original_data = read_results(filename)
  argument_class = None

  # Reconverts keys to arguments and make numpy arrays.
  data = {}
  for k, v in original_data.items():
    if len(k) == 3 and argument_class is None:
      argument_class = launch_experiments.Arguments
    elif len(k) == 4 and argument_class is None:
      argument_class = launch_experiments.OldArguments
    if argument_class is None:
      raise ValueError('Unsupported data format.')
    algorithm_data = {}
    for algorithm, (costs, correlations) in v.items():
      algorithm_data[algorithm] = Data(np.array(costs, np.float32),
                                       np.array(correlations, np.float32))
    data[argument_class(*k)] = algorithm_data

  # Get baseline values.
  defaults = argument_class()

  # Values for the x-axis.
  x_axes = collections.defaultdict(set)

  # Gather possible plots.
  for k in data:
    for field in argument_class._fields:
      d = getattr(defaults, field)
      v = getattr(k, field)
      if d != v:
        # Verify that all other fields are the same.
        for other_field in argument_class._fields:
          if other_field == field:
            continue
          if getattr(defaults, other_field) != getattr(k, other_field):
            raise NotImplementedError('Plotting multiple dimensional plots is not supported.')
        # All good.
        x_axes[field].add(v)
      if field != 'deployment_size':
        x_axes[field].add(d)

  for x_axis_label, x_axis_values in x_axes.items():
    # Sorted x values.
    x_values = sorted(x_axis_values)

    # Get y values.
    y_cost_values = collections.defaultdict(list)
    y_cost_lowers = collections.defaultdict(list)
    y_cost_uppers = collections.defaultdict(list)

    for x_axis_value in x_values:
      k = argument_class(**{x_axis_label: x_axis_value})
      u = data[k][_UPPER_BOUND].costs
      for algorithm, values in data[k].items():
        y = values.costs / (u + 1e-10)
        if algorithm == _LOWER_BOUND and y_cost_values[algorithm] and x_axis_label == 'deployment_size':
          y_cost_values[algorithm].append(y_cost_values[algorithm][-1])
          y_cost_uppers[algorithm].append(y_cost_uppers[algorithm][-1])
          y_cost_lowers[algorithm].append(y_cost_lowers[algorithm][-1])
        else:
          y_cost_values[algorithm].append(np.mean(y))
          lower, upper = errors(y)
          y_cost_uppers[algorithm].append(upper)
          y_cost_lowers[algorithm].append(lower)

    plt.figure()
    for algorithm in _ORDER_AS:
      if algorithm in _IGNORE:
        continue
      values = y_cost_values[algorithm]
      v = np.array(values, np.float32)
      u = np.array(y_cost_uppers[algorithm], np.float32)
      l = np.array(y_cost_lowers[algorithm], np.float32)
      ls = _LINESTYLE[algorithm] if algorithm in _LINESTYLE else _DEFAULT_LINESTYLE
      plt.plot(x_values, v, linestyle=ls, color=_COLORS[algorithm], lw=2, label=algorithm, marker='o', ms=8)
      plt.fill_between(x_values, l, u, facecolor=_COLORS[algorithm], alpha=.5)
    plt.xlabel(x_axis_label)
    plt.ylabel('cost')
    plt.legend()

  # Plot correlations only for the default set of values.
  y_corr_values = []
  xticks = []
  for algorithm in _ORDER_AS:
    if algorithm in _IGNORE or algorithm in (_LOWER_BOUND, _UPPER_BOUND):
      continue
    xticks.append(algorithm)
    y_corr_values.append(data[defaults][algorithm].correlations)

  plt.figure()
  x = np.arange(len(y_corr_values))
  make_nice(plt.boxplot(y_corr_values, patch_artist=True, showfliers=False))
  plt.ylabel('correlation')
  plt.xticks(x + 1, xticks)

  plt.show(block=False)
  input('Hit ENTER to close figure')
  plt.close()


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.
  parser = argparse.ArgumentParser(description='Plots experimental results')
  parser.add_argument('--input_results', action='store', required=True, help='Where the results are stored.')
  args = parser.parse_args()
  run(args.input_results)

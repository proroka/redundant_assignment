from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import matplotlib.pylab as plt
import msgpack
import msgpack_numpy
import numpy as np

import launch_experiments

Data = collections.namedtuple('Data', ['costs', 'correlations'])

_LOWER_BOUND = 'hungarian'

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


def read_results(filename):
  with open(filename, 'rb') as fp:
    return msgpack.unpackb(fp.read(), raw=False, use_list=False)


def run(filename):
  original_data = read_results(filename)

  # Reconverts keys to arguments and make numpy arrays.
  data = {}
  for k, v in original_data.items():
    algorithm_data = {}
    for algorithm, (costs, correlations) in v.items():
      algorithm_data[algorithm] = Data(np.array(costs, np.float32),
                                       np.array(correlations, np.float32))
    data[launch_experiments.Arguments(*k)] = algorithm_data

  # Get baseline values.
  defaults = launch_experiments.Arguments()

  # Values for the x-axis.
  x_axes = collections.defaultdict(set)

  # Gather possible plots.
  for k in data:
    for field in launch_experiments.Arguments._fields:
      d = getattr(defaults, field)
      v = getattr(k, field)
      if d != v:
        # Verify that all other fields are the same.
        for other_field in launch_experiments.Arguments._fields:
          if other_field == field:
            continue
          if getattr(defaults, other_field) != getattr(k, other_field):
            raise NotImplementedError('Plotting multiple dimensional plots is not supported.')
        # All good.
        x_axes[field].add(v)
      x_axes[field].add(d)

  for x_axis_label, x_axis_values in x_axes.items():
    # Sorted x values.
    x_values = sorted(x_axis_values)
    # Get y values.
    y_values = collections.defaultdict(list)
    y_stds = collections.defaultdict(list)
    for x_axis_value in x_values:
      k = launch_experiments.Arguments(**{x_axis_label: x_axis_value})
      l = data[k][_LOWER_BOUND].costs
      for algorithm, values in data[k].items():
        y = values.costs / l
        y_values[algorithm].append(np.mean(y))
        y_stds[algorithm].append(np.std(y))

    plt.figure()
    for algorithm, values in y_values.items():
      v = np.array(values, np.float32)
      s = np.array(y_stds[algorithm], np.float32)
      ls = _LINESTYLE[algorithm] if algorithm in _LINESTYLE else _DEFAULT_LINESTYLE
      plt.plot(x_values, v, linestyle=ls, color=_COLORS[algorithm], lw=2, label=algorithm, marker='o', ms=8)
      plt.fill_between(x_values, v - s, v + s, facecolor=_COLORS[algorithm], alpha=.5)
    plt.xlabel(x_axis_label)
    plt.ylabel('cost')
    plt.legend()
  plt.show()


if __name__ == '__main__':
  msgpack_numpy.patch()  # Magic.
  parser = argparse.ArgumentParser(description='Plots experimental results')
  parser.add_argument('--input_results', action='store', required=True, help='Where the results are stored.')
  args = parser.parse_args()
  run(args.input_results)

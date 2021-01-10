import numpy as np
import matplotlib.pyplot as plt

def plot_timings(n):

    filename = 'timings_' + str(n) + '.txt'
    timings = np.loadtxt(filename)

    plt.scatter(timings[:, 0], timings[:, 1])
    plt.xlabel('Number of Steps')
    plt.ylabel('Execution Time (ms)')
    plt.title('Scaling with ' + str(n) + ' Dimensions')
    plt.xlim((0, max(timings[:, 0]) + 5000))
    plt.show()


def plot_dim_timings():

    filename = 'timings_dim.txt'
    timings = np.loadtxt(filename)

    plt.scatter(timings[:, 0], timings[:, 1])
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Execution Time (ms)')
    plt.title('Dimensional Scaling with 3000 Steps')
    plt.xlim((0, max(timings[:, 0]) + 5))
    plt.show()


def plot_data_timings():

    filename = 'timings_data.txt'
    timings = np.loadtxt(filename)

    plt.scatter(timings[:, 0], timings[:, 1])
    plt.xlabel('Number of Data Points')
    plt.ylabel('Execution Time (ms)')
    plt.title('Data Scaling with 10000 Steps')
    plt.xlim((-100, max(timings[:, 0]) + 600))
    plt.show()

plot_timings(20)
plot_dim_timings()
plot_data_timings()

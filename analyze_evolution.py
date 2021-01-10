import numpy as np
import matplotlib.pyplot as plt

def calc_l2_error(n, vec):

    true_val = 5.0

    error = 0.0

    for i in range(n):
        error += (vec[i] - true_val) * (vec[i] - true_val)

    return np.sqrt(error)



def plot_evolution(n):

    filename = 'mu_evolution_' + str(n) + '.txt'
    print('Searching for file ', filename)
    evolution = np.loadtxt(filename)

    spacing = 1000
    n_samples = len(evolution) // n

    print('n_samples: ', n_samples)

    l2_evolution = np.zeros(n_samples)
    for i in range(n_samples):
        l2_evolution[i] = calc_l2_error(n, evolution[n*i:(i+1)*n])
        print('error: ', l2_evolution[i])

    steps_arr = np.arange(n_samples)
    steps_arr *= spacing

    plt.scatter(steps_arr, l2_evolution)
    plt.xlabel('Number of Steps')
    plt.ylabel(r'$\ell_2$ Norm of Error')
    plt.title('Convergence in ' + str(n) + ' Dimensions')
    plt.xlim((-100, 50100))
    plt.ylim((-0.1, 17))
    plt.show()

plot_evolution(15)

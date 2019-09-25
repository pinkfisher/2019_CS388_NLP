# Plot the CRF experimentation data

import matplotlib.pyplot as plt
import numpy as np


def plot():
    f1 = np.array(
        [74.51, 83.25, 84.80, 85.33, 85.74, 85.93, 85.94, 86.02,
         86.04, 86.09, 86.23])
    f1_shuffle = np.array(
        [78.47, 85.34, 86.52, 86.57, 86.94, 86.69, 86.99]
    )
    n = np.array(range(0, 55, 5))

    plt.plot(n, f1, "-*", label='CRF Viterbi Algorithm')
    plt.plot(n[:7], f1_shuffle, "-*", label='CRF Viterbi Algorithm (shuffle)')
    plt.title('F1 Score vs. Number of Epoches')
    plt.xlabel('number of epoches')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    plot()
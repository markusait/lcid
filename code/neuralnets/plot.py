import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_rate(x_y_data, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    for x, y, label in x_y_data:
        x_series = np.arange(0, x, 1)
        y_series = y
        ax.plot(x_series, y_series, label=f"Learning Rate: {label}")

    ax.set(xlabel=xlabel, ylabel=ylabel,
        title=title)
    ax.grid()

    # fig.savefig("test.png")
    plt.legend(loc='best')
    plt.show()

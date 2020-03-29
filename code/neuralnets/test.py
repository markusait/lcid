import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from neuralnets import mnist_loader, network, plot


# This is how we define multiple networks for comparison
networks = [
    {
        'sizes': [784, 30, 10],
        'epochs': 2,
        'mini_batch_size': 10,
        'learning_rate': 2.0,
    },
]
""""
    {
        'sizes': [784, 30, 10],
        'epochs': 25,
        'mini_batch_size': 10,
        'learning_rate': 2.5,
    },
    {
        'sizes': [784, 10],
        'epochs': 25,
        'mini_batch_size': 10,
        'learning_rate': 2.5,
    },
"""
print("Plotting the networks now, this may take a while")


def plot_networks(networks):
    plot_data = []
    for network_data in networks:
        print("network data", network_data)
        # Reloading on each iteration because of EOF issues with training_data
        training_data, _, test_data = mnist_loader.load_data_wrapper()
        net = network.Network(network_data['sizes'])
        cr = net.SGD(training_data=training_data, epochs=network_data['epochs'], mini_batch_size=network_data[
                     'mini_batch_size'], eta=network_data['learning_rate'], test_data=test_data)
        plot_data.append(
            (network_data['epochs'], cr, network_data['learning_rate']))

    # Now we plot the classification results of all network
    plot.plot_learning_rate(plot_data, xlabel="epochs", ylabel="Correct Classifications %",
         title='Training Model output')


plot_networks(networks)
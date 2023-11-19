import matplotlib.pyplot as plt

# Plot the results
def plot_trajectory(dataset):
    t = dataset["time"]
    wolves = dataset["wolves"]
    rabbits = dataset["rabbits"]
    fig, ax = plt.subplots(1,1)
    ax.plot(t, rabbits, label=rabbits.name, color='blue')
    ax.plot(t, wolves, label=wolves.name, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title('Lotka-Volterra Predator-Prey Model')
    ax.legend()
    ax.grid(True)

    return fig
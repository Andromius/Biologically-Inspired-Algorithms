import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=0)

def schwefel(x):
    n = len(x)
    return 418.9829 * n - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def griewangk(x):
    sum_term = sum(xi**2 for xi in x) / 4000
    prod_term = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return 1 + sum_term - prod_term

def levy(x):
    x = np.array(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2), axis=0)
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3

def michalewicz(x, m=10):
    return -sum(np.sin(xi) * (np.sin(i * xi**2 / np.pi))**(2*m) for i, xi in enumerate(x, 1))

def zakharov(x):
    n = len(x)
    term1 = sum(x_i ** 2 for x_i in x)
    term2 = sum(0.5 * (i + 1) * x[i] for i in range(n)) ** 2
    term3 = sum(0.5 * (i + 1) * x[i] for i in range(n)) ** 4
    return term1 + term2 + term3

def ackley(x):
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e

def update_frame(
    i: int,
    xy_data: list[np.array],
    z_data: list[np.array],
    scat,
    ax,
):
    scat[0].remove()
    scat[0] = ax[0].scatter(
        xy_data[i][:, 0], xy_data[i][:, 1], z_data[i], c="red"
    )


def render_anim(
    surface_X: np.array,
    surface_Y: np.array,
    surface_Z: np.array,
    xy_data: list[np.array],
    z_data: list[np.array],
):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        surface_X,
        surface_Y,
        surface_Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.6,
    )
    # render first frame
    scat = ax.scatter(xy_data[0][:, 0], xy_data[0][:, 1], z_data[0], c="red")

    animation = FuncAnimation(
        fig,
        update_frame,
        len(xy_data),
        fargs=(xy_data, z_data, [scat], [ax]),
        interval=500, repeat=False
    )
    plt.show()


def render_graph(
    surface_X: np.array,
    surface_Y: np.array,
    surface_Z: np.array,
):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        surface_X,
        surface_Y,
        surface_Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.6,
    )
    plt.show()


def make_surface(
    min: float,
    max: float,
    function: callable,
    step: float,
):
    X = np.arange(min, max, step)
    Y = np.arange(min, max, step)
    X, Y = np.meshgrid(X, Y)
    Z = function(np.array([X, Y]))
    return X, Y, Z

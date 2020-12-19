"""
@Authors:
Fernando Garcia Sanz - 970718-0312
Gustavo Teodoro Döhler Beck - 940218-0195
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def main():
    np.random.seed(1337)
    model = torch.load("neural-network-1.pth")
    y = np.linspace(0, 1.5, 100)
    w = np.linspace(-np.pi, np.pi, 100)
    mat = np.zeros((len(y), len(w)))
    for idx, i in enumerate(y):
        for jdx, j in enumerate(w):
            state = torch.tensor((0, i, 0, 0, j, 0, 0, 0), dtype=torch.float32)
            mat[idx, jdx] = model(state).max(0)[1].item()  # Argmax
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    y, w = np.meshgrid(y, w, sparse=False, indexing='ij')
    # Plot the surface.
    ax.plot_surface(y, w, mat, cmap='viridis', edgecolor='none')
    ax.set_xlabel('height (y)')
    ax.set_ylabel('angle (w)')
    ax.set_zlabel(r'$\argmax_a Q(s(y,w),a)$')
    ax.set_title(r'$\argmax_a Q(s(y,w),a)$')
    plt.savefig("argmax.png")
    plt.show()

    y = np.linspace(0, 1.5, 100)
    w = np.linspace(-np.pi, np.pi, 100)
    mat = np.zeros((len(y), len(w)))
    for idx, i in enumerate(y):
        for jdx, j in enumerate(w):
            state = torch.tensor((0, i, 0, 0, j, 0, 0, 0), dtype=torch.float32)
            mat[idx, jdx] = model(state).max(0)[0].item()  # Max
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    y, w = np.meshgrid(y, w, sparse=False, indexing='ij')
    # Plot the surface.
    ax.plot_surface(y, w, mat, cmap=cm.coolwarm, edgecolor='none')
    ax.set_xlabel('height (y)')
    ax.set_ylabel('angle (w)')
    ax.set_zlabel(r'$\max_a Q(s(y,w),a)$')
    ax.set_title(r'$\max_a Q(s(y,w),a)$')
    plt.savefig("max.png")
    plt.show()


if __name__ == "__main__":
    main()

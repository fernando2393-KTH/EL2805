import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def main():
    np.random.seed(1337)
    model_actor = torch.load("neural-network-3-actor.pth", map_location=torch.device('cpu'))
    model_critic = torch.load("neural-network-3-critic.pth", map_location=torch.device('cpu'))
    y = np.linspace(0, 1.5, 100)
    w = np.linspace(-np.pi, np.pi, 100)
    mat = np.zeros((len(y), len(w)))
    for idx, i in enumerate(y):
        for jdx, j in enumerate(w):
            state = torch.tensor((0, i, 0, 0, j, 0, 0, 0), dtype=torch.float32).reshape(1, -1)
            mat[idx, jdx] = model_critic(state).item()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    y, w = np.meshgrid(y, w, sparse=False, indexing='ij')
    # Plot the surface.
    ax.plot_surface(y, w, mat, cmap=cm.coolwarm, edgecolor='none')
    ax.set_xlabel('height (y)')
    ax.set_ylabel('angle (w)')
    ax.set_zlabel(r'$V_w(s(y,w))$')
    ax.set_title(r'$V_w(s(y,w))$')
    plt.show()

    y = np.linspace(0, 1.5, 100)
    w = np.linspace(-np.pi, np.pi, 100)
    mat_actor = np.zeros((len(y), len(w)))
    for idx, i in enumerate(y):
        for jdx, j in enumerate(w):
            state = torch.tensor((0, i, 0, 0, j, 0, 0, 0), dtype=torch.float32)
            mat_actor[idx, jdx] = model_actor(state)[0][1].item()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    y, w = np.meshgrid(y, w, sparse=False, indexing='ij')
    # Plot the surface.
    ax.plot_surface(y, w, mat_actor, cmap='viridis', edgecolor='none')
    ax.set_xlabel('height (y)')
    ax.set_ylabel('angle (w)')
    ax.set_zlabel('Engine Direction')
    ax.set_title(r'$\mu_\theta(s(y,w))_2$')
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

def plotBifurcationDiagramA():
    data = np.load('./data/bifurcation_diagram_A.npy')
    a_values, T_max_values, CR_values, grad_penalties, excess_penalties, J_values, dJ_values = np.unstack(data)

    # Plot the  conversion rates and max temperatures
    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    color2 = "tab:red"

    # Left y-axis: conversion
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel("CO conversion", color=color1)
    l1 = ax1.plot(a_values, CR_values / 100.0, color=color1, label="Conversion")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.0, 1.05)  # nice [0,1] scale for conversion

    # Right y-axis: max temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$T_{\max}$ [K]", color=color2)
    l2 = ax2.plot(a_values, T_max_values, color=color2, label=r"$T_{\max}$")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    ax1.set_title(r"CO Conversion and hot-spot vs $a$")
    fig.tight_layout()
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Also plot the objective values, gradient penalties, excess penalties, J and dJ
    plt.figure()
    plt.semilogy(a_values, CR_values, label=r"Conversion Rate $(\%)$")
    plt.semilogy(a_values, grad_penalties, label='Gradient Penalty')
    plt.semilogy(a_values, excess_penalties, label='Excess Penalty')
    plt.xlabel(r"$a$")
    plt.legend()

    # Determine the optimal objective function
    plt.figure()
    CR_loss = 100.0 - CR_values
    gamma_values = [0.0, 1000.0, 10000.0]
    for gamma in gamma_values:
        plt.semilogy(a_values, CR_loss + gamma * excess_penalties, label=rf"$\gamma = {gamma}$")
    plt.xlabel(r"$a$")
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    plotBifurcationDiagramA()
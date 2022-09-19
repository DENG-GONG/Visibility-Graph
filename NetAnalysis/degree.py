import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
def draw_degree_frequency_distribution(G:nx.Graph,save_to='fig_results/',need_show=True,**kwargs):
    degree_list=nx.degree_histogram(G)
    x=range(len(degree_list))
    y=[z/float(sum(degree_list))for z in degree_list]
    plt.loglog(x, y, color="blue", linewidth=2,linestyle=':',**kwargs)
    plt.savefig(save_to+'degree_frequency')
    if need_show:
        plt.show()
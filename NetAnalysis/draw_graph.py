from DaChuang2021.NetAnalysis.as_Graph import *
from DaChuang2021.visual_graph_algorithm import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
def Search_save_to(save_to):
    if not os.path.exists(save_to):
        os.makedirs(save_to)
def draw_VG(data,need_show=True,save_to='fig_results/',need_degree=True,**kwargs):
    Search_save_to(save_to)
    plt.figure(dpi=300)
    VGdata=VG(data)
    VGdata.to_csv(save_to+'VGdata.csv')
    G = make_VGgraph(VGdata)
    nx.draw(G, **kwargs)
    plt.savefig(save_to + 'VVGnet.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(G, save_to + 'VVG', need_show, **kwargs)
    return G
def draw_HVG(data,need_show=True,save_to='fig_results/',need_degree=True,**kwargs):
    Search_save_to(save_to)
    plt.figure(dpi=300)
    HVGdata=HVG(data)
    HVGdata.to_csv(save_to+'HVG.csv')
    G = make_HVGgraph(HVGdata)
    nx.draw(G, **kwargs)
    plt.savefig(save_to + 'HVGnet.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(G, save_to + 'VVG', need_show, **kwargs)
    return G
def draw_LPVG(data,N,VGnodecolor='g',VGedgecolor='black',LPVGnodecolor='r',LPVGedgecolor='b',save_to='fig_results/',need_show=True, need_degree=True,**kwargs):
    Search_save_to(save_to)
    plt.figure(dpi=300)
    LPVGdata=LPVG(data, N)
    VGdf = LPVGdata.loc[:, ['VGi', 'VGj']].dropna()
    LPVGdf = LPVGdata.loc[:, ['LPVGi', 'LPVGj']].dropna()
    # print(df2['LPVGi'])
    G = make_LPVGgraph(LPVGdf, False, VGdf)
    node_colors = [LPVGnodecolor if "LPVG" in G.nodes[u]["VGtype"] else VGnodecolor for u in G.nodes]
    edge_colors = [LPVGedgecolor if "LPVG" in G.edges[u]["VGtype"] else VGedgecolor for u in G.edges]
    nx.draw(G,pos=nx.circular_layout(G), node_color=node_colors,edge_color=edge_colors,**kwargs)
    plt.savefig(save_to + 'LPVG.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(G,save_to+'LPVG',need_show,**kwargs)

    return G
def draw_LPHVG(data,N,VGnodecolor='g',VGedgecolor='black',LPVGnodecolor='r',LPVGedgecolor='b',save_to='fig_results/',need_show=True, need_degree=True,**kwargs):
    Search_save_to(save_to)
    plt.figure(dpi=300)
    HVGdf = LPHVG(data, N).loc[:, ['HVGi', 'HVGj']].dropna()
    LPHVGdf = LPHVG(data, N).loc[:, ['LPHVGi', 'LPHVGj']].dropna()
    # print(df2['LPVGi'])
    G = make_LPHVGgraph(LPHVGdf, False, HVGdf)
    node_colors = [LPVGnodecolor if "LPHVG" in G.nodes[u]["VGtype"] else VGnodecolor for u in G.nodes]
    edge_colors = [LPVGedgecolor if "LPHVG" in G.edges[u]["VGtype"] else VGedgecolor for u in G.edges]
    nx.draw(G,pos=nx.circular_layout(G), node_color=node_colors,edge_color=edge_colors,**kwargs)
    plt.savefig(save_to + 'LPHVG.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(G,save_to+'LPHVG',need_show,**kwargs)
    return G
def draw_MlLPVG(data,N,Core,VGnodecolor='g',VGedgecolor='black',LPVGnodecolor='r',LPVGedgecolor='b',save_to='fig_results/',need_show=True, need_degree=True,**kwargs):
    Search_save_to(save_to)
    def __draw_LPVG(df:pd.DataFrame):
        plt.figure(dpi=300)
        VGdf = df.loc[:, ['VGi', 'VGj']].dropna()
        LPVGdf = df.loc[:, ['LPVGi', 'LPVGj']].dropna()
        G = make_LPVGgraph(LPVGdf, False, VGdf)
        node_colors = [LPVGnodecolor if "LPVG" in G.nodes[u]["VGtype"] else VGnodecolor for u in G.nodes]
        edge_colors = [LPVGedgecolor if "LPVG" in G.edges[u]["VGtype"] else VGedgecolor for u in G.edges]
        nx.draw(G,pos=nx.circular_layout(G), node_color=node_colors,edge_color=edge_colors,**kwargs)
        return G
    MlLPVGresults=MlLPVG(data,N,Core)
    i=0
    for LPVGresult in MlLPVGresults['LPVGresults']:
        G=__draw_LPVG(LPVGresult)
        plt.savefig(save_to+'dim%d.LPVG.png'%i)
        if need_show:
            plt.show()
        if need_degree:
            draw_degree_frequency_distribution(G, save_to + 'dim%d.LPVG.png'%i, need_show, **kwargs)
        i+=1
    P1,P2=make_MlLPVG_projectionNet(MlLPVGresults)

    P1node_colors = [LPVGnodecolor if "LPVG" in P1.nodes[u]["VGtype"] else VGnodecolor for u in P1.nodes]
    P1edge_colors = [LPVGedgecolor if "LPVG" in P1.edges[u]["VGtype"] else VGedgecolor for u in P1.edges]
    plt.figure(dpi=300)
    nx.draw(P1, pos=nx.circular_layout(P1),node_color=P1node_colors, edge_color=P1edge_colors,**kwargs)
    plt.savefig(save_to + 'P1.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(P1,save_to+'P1',need_show,**kwargs)
    P2node_colors = [LPVGnodecolor if "LPVG" in P2.nodes[u]["VGtype"] else VGnodecolor for u in P2.nodes]
    P2edge_colors = [LPVGedgecolor if "LPVG" in P2.edges[u]["VGtype"] else VGedgecolor for u in P2.edges]
    plt.figure(dpi=300)
    nx.draw(P2,pos=nx.circular_layout(P2), node_color=P2node_colors,edge_color=P2edge_colors,**kwargs)
    plt.savefig(save_to + 'P2.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(P2,save_to+'P2',need_show,**kwargs)
    return P1,P2
def draw_MlLPHVG(data, N, Core, HVGnodecolor='g', HVGedgecolor='black', LPHVGnodecolor='r', LPHVGedgecolor='b', save_to='fig_results/', need_show=True, need_degree=True,**kwargs):
    Search_save_to(save_to)
    def __draw_LPHVG(df:pd.DataFrame):
        plt.figure(dpi=300)
        HVGdf = df.loc[:, ['HVGi', 'HVGj']].dropna()
        # print(HVGdf)
        LPHVGdf = df.loc[:, ['LPHVGi', 'LPHVGj']].dropna()
        G = make_LPHVGgraph(LPHVGdf, False, HVGdf)
        # print([G.nodes[u] for u in G.nodes])
        node_colors = [LPHVGnodecolor if "LPHVG" in G.nodes[u]["VGtype"] else HVGnodecolor for u in G.nodes]
        edge_colors = [LPHVGedgecolor if "LPHVG" in G.edges[u]["VGtype"] else HVGedgecolor for u in G.edges]
        nx.draw(G,pos=nx.circular_layout(G), node_color=node_colors,edge_color=edge_colors,**kwargs)
        return G

    MlLPHVGresults=MlLPHVG(data,N,Core)
    i=0
    for LPHVGresult in MlLPHVGresults['LPHVGresults']:
        G=__draw_LPHVG(LPHVGresult)
        plt.savefig(save_to+'dim%d.LPHVG.png'%i)
        if need_show:
            plt.show()
        if need_degree:
            draw_degree_frequency_distribution(G, save_to + 'dim%d.LPHVG.png'%i, need_show, **kwargs)
        i+=1
    P1,P2=make_MlLPHVG_projectionNet(MlLPHVGresults)

    P1node_colors = [LPHVGnodecolor if "LPHVG" in P1.nodes[u]["VGtype"] else HVGnodecolor for u in P1.nodes]
    P1edge_colors = [LPHVGedgecolor if "LPHVG" in P1.edges[u]["VGtype"] else HVGedgecolor for u in P1.edges]
    plt.figure(dpi=300)
    nx.draw(P1, pos=nx.circular_layout(P1),node_color=P1node_colors, edge_color=P1edge_colors,**kwargs)
    plt.savefig(save_to + 'P1.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(P1,save_to+'P1',need_show,**kwargs)
    P2node_colors = [LPHVGnodecolor if "LPHVG" in P2.nodes[u]["VGtype"] else HVGnodecolor for u in P2.nodes]
    P2edge_colors = [LPHVGedgecolor if "LPHVG" in P2.edges[u]["VGtype"] else HVGedgecolor for u in P2.edges]
    plt.figure(dpi=300)
    nx.draw(P2,pos=nx.circular_layout(P2), node_color=P2node_colors,edge_color=P2edge_colors,**kwargs)
    plt.savefig(save_to + 'P2.png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(P2,save_to+'P2',need_show,**kwargs)
    return P1,P2
def draw_VVG(data,Core,save_to='fig_results/', need_show=True, need_degree=True,**kwargs):
    Search_save_to(save_to)
    plt.figure(dpi=300)
    G=make_VVGgraph(VVG(data,Core,save_to))
    nx.draw(G,**kwargs)
    plt.savefig(save_to+'VVGnet,png')
    if need_show:
        plt.show()
    if need_degree:
        draw_degree_frequency_distribution(G,save_to+'VVG',need_show,**kwargs)
    return G
def draw_degree_frequency_distribution(G:nx.Graph,save_to='fig_results/',need_show=True,**kwargs):
    plt.cla()
    plt.close()
    plt.figure()
    degree_list=nx.degree_histogram(G)
    print(degree_list)
    plt.hist(degree_list,density=True,stacked=True,log=False)
    # x=range(len(degree_list))
    # y=[z/float(sum(degree_list))for z in degree_list]
    # plt.loglog(x, y, color="blue", linewidth=2,linestyle=':')
    plt.savefig(save_to+'degree_frequency.png')
    if need_show:
        plt.show()


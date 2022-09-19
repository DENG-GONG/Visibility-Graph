import pandas as pd
import numpy  as np
import networkx as nx

def make_VGgraph(df:pd.DataFrame):
    """

    :param df: 需要清洗过后的只含VGi,VGj部分的df
    :return:
    """
    VGgraph = nx.Graph(VGtype='VG')
    VGgraph.add_nodes_from([i for i in df['VGi']], VGtype='VG')
    VGgraph.add_nodes_from([j for j in df['VGj']], VGtype='VG')
    VGgraph.add_edges_from([[i, j] for i, j in zip(df['VGi'], df['VGj'])], VGtype='VG')
    return VGgraph
def make_HVGgraph(df:pd.DataFrame):
    """

    :param df: 需要清洗过后的只含VGi,VGj部分的df
    :return:
    """
    HVGgraph = nx.Graph(VGtype='HVG')
    HVGgraph.add_nodes_from([i for i in df['HVGi']], VGtype='HVG')
    HVGgraph.add_nodes_from([j for j in df['HVGj']], VGtype='HVG')
    # print([[i, j] for i, j in zip(df['HVGi'], df['HVGj'])])
    HVGgraph.add_edges_from([[i, j] for i, j in zip(df['HVGi'], df['HVGj'])], VGtype='HVG')
    # print(HVGgraph.edges)
    return HVGgraph
def make_VVGgraph(df:pd.DataFrame):
    """

    :param df: 需要清洗过后的只含VGi,VGj部分的df
    :return:
    """
    VVGgraph = nx.DiGraph(VGtype='VVG')
    # VVGgraph.a ([i for i in df['VVGi']], VGtype='VVGout')
    # VVGgraph.add_nodes_from([j for j in df['VVGj']], VGtype='VVGint')
    VVGgraph.add_edges_from([[i, j] for i, j in zip(df['VVGi'], df['VVGj'])], VGtype='VVG')
    return VVGgraph
def make_LPVGgraph(only_LPVGdf:pd.DataFrame, is_only=False,VGdf=None ):
    """
    返回不包含VG的LPVG图,或包含VG的LPVG图,支持MsLPVG
    :param only_LPVGdf: 清洗后的只含LPVG的两列的df切片
    :param is_only: 设定返回的类型
    :param VGdf: 清洗后只含VG的两列df切片
    :return:
    """
    only_LPVGgraph = nx.Graph(VGtype='only_LPVG')
    only_LPVGgraph.add_nodes_from([i for i in only_LPVGdf['LPVGi']], VGtype='LPVG')
    only_LPVGgraph.add_nodes_from([j for j in only_LPVGdf['LPVGj']], VGtype='LPVG')
    only_LPVGgraph.add_edges_from([[i, j] for i, j in zip(only_LPVGdf['LPVGi'], only_LPVGdf['LPVGj'])], VGtype='LPVG')
    if is_only:
        return only_LPVGgraph
    else:
        VGgraph=make_VGgraph(VGdf)
        LPVGgraph=nx.Graph(VGtype='LPVG')
        LPVGgraph.add_nodes_from(VGgraph.nodes,VGtype='VG')
        LPVGgraph.add_nodes_from(only_LPVGgraph.nodes,VGtype='LPVG')
        LPVGgraph.add_edges_from(VGgraph.edges,VGtype='VG')
        LPVGgraph.add_edges_from(only_LPVGgraph.edges,VGtype='LPVG')
        return LPVGgraph
def make_LPHVGgraph(only_LPHVGdf:pd.DataFrame, is_only=False, HVGdf=None):
    """
    返回不包含VG的LPVG图,或包含VG的LPVG图,支持MsLPHVG
    :param only_LPHVGdf: 清洗后的只含LPVG的两列的df切片
    :param is_only: 设定返回的类型
    :param HVGdf: 清洗后只含VG的两列df切片
    :return:
    """
    only_LPHVGgraph = nx.Graph(VGtype='only_LPHVG')
    only_LPHVGgraph.add_nodes_from([i for i in only_LPHVGdf['LPHVGi']], VGtype='LPHVG')
    only_LPHVGgraph.add_nodes_from([j for j in only_LPHVGdf['LPHVGj']], VGtype='LPHVG')
    only_LPHVGgraph.add_edges_from([[i, j] for i, j in zip(only_LPHVGdf['LPHVGi'], only_LPHVGdf['LPHVGj'])], VGtype='LPHVG')
    if is_only:
        return only_LPHVGgraph
    else:
        HVGgraph=make_HVGgraph(HVGdf)
        LPHVGgraph=nx.Graph(VGtype='LPHVG')
        LPHVGgraph.add_nodes_from(HVGgraph.nodes,VGtype='HVG')
        LPHVGgraph.add_nodes_from(only_LPHVGgraph.nodes,VGtype='LPHVG')
        LPHVGgraph.add_edges_from(HVGgraph.edges,VGtype='HVG')
        LPHVGgraph.add_edges_from(only_LPHVGgraph.edges,VGtype='LPHVG')
        return LPHVGgraph
def make_MlLPVG_projectionNet(dict:{}):
    """
    :param dict: 必须用MlLPVG返回的dict
    :return:
    """
    P1=nx.Graph(nettype='andNet')
    P1.add_nodes_from([i for i in dict['P1V']['VGi']],VGtype='VG')
    P1.add_nodes_from([j for j in dict['P1V']['VGj']], VGtype='VG')
    P1.add_nodes_from([i for i in dict['P1L']['LPVGi']], VGtype='LPVG')
    P1.add_nodes_from([j for j in dict['P1L']['LPVGj']], VGtype='LPVG')
    # 添加P1边
    P1.add_edges_from([[i,j]for i,j in zip(dict['P1V']['VGi'],dict['P1V']['VGj'])],VGtype='VG')
    P1.add_edges_from([[i,j]for i,j in zip(dict['P1L']['LPVGi'],dict['P1L']['LPVGj'])],VGtype='LPVG')
    # 生成P2
    P2 = nx.Graph(nettyoe='orNet')
    P2.add_nodes_from([i for i in dict['P2V']['VGi']], VGtype='VG')
    P2.add_nodes_from([j for j in dict['P2V']['VGj']], VGtype='VG')
    P2.add_nodes_from([i for i in dict['P2L']['LPVGi']], VGtype='LPVG')
    P2.add_nodes_from([j for j in dict['P2L']['LPVGj']], VGtype='LPVG')
    # 添加P2边
    P2.add_edges_from([[i, j] for i, j in zip(dict['P2V']['VGi'], dict['P2V']['VGj'])], VGtype='VG')
    P2.add_edges_from([[i, j] for i, j in zip(dict['P2L']['LPVGi'], dict['P2L']['LPVGj'])], VGtype='LPVG')
    return P1,P2
def make_MlLPHVG_projectionNet(dict:{}):
    """
        :param dict: 必须用MlLPVG返回的dict
        :return:
        """
    P1 = nx.Graph(nettype='andNet')
    P1.add_nodes_from([i for i in dict['P1V']['HVGi']], VGtype='HVG')
    P1.add_nodes_from([j for j in dict['P1V']['HVGj']], VGtype='HVG')
    P1.add_nodes_from([i for i in dict['P1L']['LPHVGi']], VGtype='LPHVG')
    P1.add_nodes_from([j for j in dict['P1L']['LPHVGj']], VGtype='LPHVG')
    # 添加P1边
    P1.add_edges_from([[i, j] for i, j in zip(dict['P1V']['HVGi'], dict['P1V']['HVGj'])], VGtype='HVG')
    P1.add_edges_from([[i, j] for i, j in zip(dict['P1L']['LPHVGi'], dict['P1L']['LPHVGj'])], VGtype='LPHVG')
    # 生成P2
    P2 = nx.Graph(nettyoe='orNet')
    P2.add_nodes_from([i for i in dict['P2V']['HVGi']], VGtype='HVG')
    P2.add_nodes_from([j for j in dict['P2V']['HVGj']], VGtype='HVG')
    P2.add_nodes_from([i for i in dict['P2L']['LPHVGi']], VGtype='LPHVG')
    P2.add_nodes_from([j for j in dict['P2L']['LPHVGj']], VGtype='LPHVG')
    # 添加P2边
    P2.add_edges_from([[i, j] for i, j in zip(dict['P2V']['HVGi'], dict['P2V']['HVGj'])], VGtype='HVG')
    P2.add_edges_from([[i, j] for i, j in zip(dict['P2L']['LPHVGi'], dict['P2L']['LPHVGj'])], VGtype='LPHVG')
    return P1, P2

# if __name__ == '__main__':
#     q = np.array([0.9, 0.5, 0.4, 0.8, 0.9, 0.5, 0.4, 0.8, 0.9, 0.5, 0.4, 0.8, 0.9])
#     df = LPVG(q, 1).loc[:, ['VGi', 'VGj']]
#     # VGgraph=nx.Graph(VGtype='VG')
#     # VGgraph.add_nodes_from([i for i in df['VGi']], VGtype='VG')
#     # VGgraph.add_nodes_from([j for j in df['VGj']], VGtype='VG')
#     # VGgraph.add_edges_from([[i, j] for i, j in zip(df['VGi'], df['VGj'])], VGtype='VG')

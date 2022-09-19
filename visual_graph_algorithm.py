import numpy as np
import pandas as pd
import multiprocessing as mlp
def to_array(df:pd.DataFrame):
    return np.asarray(df)

def iterfun(n):
    """
    生成i,j其中i for 0:n,j in i+1:n
    :param n:
    :return:
    """
    for i in range(n):
        for j in range(i+1,n):
            yield i,j
def VG(UTS):
    y = np.array(UTS)
    result=[]
    # 遍历,获得连接两点的横坐标
    # 控制ta移动
    for ta,tb in iterfun(len(UTS)):
        df=pd.DataFrame([[None,None]],columns=['VGi','VGj'])
        ya = y[ta]
        yb = y[tb]
        tc = np.array(range(ta + 1, tb))
        yc = y[(ta + 1):tb]
        # 如果相邻，或者值1大于值2，则记录下来
        # 并打印相连的两点的坐标用(ta,tb)表示
        if tb - ta == 1 or min((yb - yc) / (tb - tc)) > (yb - ya) / (tb - ta):
            df.iloc[0,0:2]=[ta,tb]
            # print(only_LPHVGdf)
            result.append(df)
    return pd.concat(result,0)
def HVG(UTS):
    y = np.array(UTS)
    result=[]
    # 遍历,获得连接两点的横坐标
    # 控制ta移动
    for ta,tb in iterfun(len(UTS)):
        df=pd.DataFrame([[None,None]],columns=['HVGi','HVGj'])
        ya = y[ta]
        yb = y[tb]
        yc = y[(ta + 1):tb]
        if tb - ta == 1 or min([ya,yb])>max(yc):
            df.iloc[0,0:2]=[ta,tb]
            result.append(df)
    return pd.concat(result,0)
def LPVG(UTS,N):
    y = np.array(UTS)
    result=[]
    # 遍历,获得连接两点的横坐标
    # 控制ta移动
    for ta,tb in iterfun(len(UTS)):
        df=pd.DataFrame([[None,None,None,None]],columns=['VGi','VGj','LPVGi','LPVGj'])
        ya = y[ta]
        yb = y[tb]
        tc = np.array(range(ta + 1, tb))
        yc = y[(ta + 1):tb]
        # 如果相邻，或者值1大于值2，则记录下来
        # 并打印相连的两点的坐标用(ta,tb)表示
        if tb - ta == 1 or min((yb - yc) / (tb - tc)) > (yb - ya) / (tb - ta):
            df.iloc[0,0:2]=[ta,tb]
            # print(only_LPHVGdf)
            result.append(df)
            # 计算n的值
        n = sum((yb - yc) / (tb - tc) <= (yb - ya) / (tb - ta))
        # print(n)
        # print(n<=N and n>0)
        # 如果0<n<=N,则添加两点
        if n > 0 and n <= N:
            # print(n)
            df.iloc[0,2:4]=[ta,tb]
            result.append(df)
    return pd.concat(result,0)
def LPHVG(UTS,N):
    y = np.array(UTS)
    result=[]
    # 遍历,获得连接两点的横坐标
    # 控制ta移动
    for ta,tb in iterfun(len(UTS)):
        df=pd.DataFrame([[None,None,None,None]],columns=['HVGi','HVGj','LPHVGi','LPHVGj'])
        ya = y[ta]
        yb = y[tb]
        yc = y[(ta + 1):tb]
        # 如果相邻，或者值1大于值2，则记录下来
        # 并打印相连的两点的坐标用(ta,tb)表示
        if tb - ta == 1 or min([ya,yb])>max(yc):
            df.iloc[0,0:2]=[ta,tb]
            result.append(df)
            # 计算n的值
        n = sum(yc>=min([ya,yb]))
        # 如果0<n<=N,则添加两点
        if n > 0 and n <= N:
            df.iloc[0,2:4]=[ta,tb]
            result.append(df)
    return pd.concat(result,0)
def mlp_VVG(MTS,start):
    df = []
    def A_to_any(index_any):
        if index_any != start[0]:
            # print(index_any,start[0])
            array_any = MTS[:,index_any]
            # print(array_A,array_any)
            dot_AB = np.dot(array_A, array_any)
            return(dot_AB / A_length)
        else:
            return(A_length)
    array_A=MTS[:,start[0]]
    # print(array_A)
    A_length=np.linalg.norm(array_A)
    for index_B in range(MTS.shape[1]):
        if start[0] != index_B:
            flag=True
            A_to_B=A_to_any(index_B)
            for index_C in range(start[0]+1,index_B):
                A_to_C=A_to_any(index_C)
                if A_to_C>= A_to_B+(A_length-A_to_B)*(index_B-start[0])/(index_B-index_C):
                    flag=False
                    break
            if not flag:
                continue
            else:
                df.append(pd.DataFrame([[start[0],index_B]],columns=['VVGi','VVGj']))
    try:
        return pd.concat(df)
    except:
        pass
def VVG(MTS:np.array,Core:int,save_to):

    """

    :param MTS:m维n长的MTS.以[[1..m]*n]输入或n*m的矩阵输入
    :param Core: 要保留的核心数
    :return: VVG的要连边的i,j节点对组成的df
    """

    if not isinstance(MTS,np.ndarray):
        try:
            MTS= np.asarray(MTS)
        except:
            raise Exception( "必须为可转为np.array的类型")
    core_num = mlp.cpu_count() - Core
    if core_num <= 0:
        raise Warning('Core为要保留的核心数,过大则只调用1核心运行')
        core_num = 1
    pool = mlp.Pool(core_num)
    VVGresults=[pool.apply(mlp_VVG,args=(MTS,[i],))for i in range(MTS.shape[1]) ]

    pool.close()
    pool.join()
    VVGresults=pd.concat(VVGresults,axis=0)
    VVGresults.to_csv(save_to+'VVG.csv')
    return VVGresults

def MlLPVG(MTS,N,Core):
    results=[]
    pool=mlp.Pool(mlp.cpu_count()-Core)
    for UTS in MTS:
        # print(UTS[0:5])
        results.append(pool.apply(LPVG,args=(UTS,N)))
    pool.close()
    pool.join()
    # print(results[0].keys())
    # print(results[0])
    # print(results[1].iloc[0:2,:].dropna())
    for i in range(len(results)):
        V=results[i].iloc[:,0:2].dropna()
        L=results[i].iloc[:,2:4].dropna()
        if i == 0:
            P1V = V
            P1L = L
            P2V = V
            P2L = L
            # print(P1V)
        else:
            P1V = pd.merge(P1V, V, on=['VGi', 'VGj'])
            P1L = pd.merge(P1L, L, on=['LPVGi', 'LPVGj'])
            P2V = pd.merge(P2V, V, how='outer', on=['VGi', 'VGj'])
            P2L = pd.merge(P2L, L, how='outer', on=['LPVGi', 'LPVGj'])
    P1V.astype(str)
    P1L.astype(str)
    P2V.astype(str)
    P2L.astype(str)
    return {'LPVGresults':results,'P1V':P1V,'P1L':P1L,'P2V':P2V,'P2L':P2L}
    # print('交集网络P1V:')
    # print(P1V)
    # print('交集P1L:')
    # print(P1L)
    # print('并集P2V:')
    # print(P2V)
    # print('并集P2L:')
    # print(P2L)
def MlLPHVG(MTS,N,Core):
    results = []
    pool = mlp.Pool(mlp.cpu_count() - Core)
    for UTS in MTS:
        # print(UTS[0:5])
        results.append(pool.apply(LPHVG, args=(UTS, N)))
    pool.close()
    pool.join()
    # print(results[0].keys())
    # print(results[0])
    # print(results[1].iloc[0:2,:].dropna())
    for i in range(len(results)):
        V = results[i].iloc[:, 0:2].dropna()
        L = results[i].iloc[:, 2:4].dropna()
        if i == 0:
            P1V = V
            P1L = L
            P2V = V
            P2L = L
            # print(P1V)
        else:
            P1V = pd.merge(P1V, V, on=['HVGi', 'HVGj'])
            P1L = pd.merge(P1L, L, on=['LPHVGi', 'LPHVGj'])
            P2V = pd.merge(P2V, V, how='outer', on=['HVGi', 'HVGj'])
            P2L = pd.merge(P2L, L, how='outer', on=['LPHVGi', 'LPHVGj'])
    P1V.astype(str)
    P1L.astype(str)
    P2V.astype(str)
    P2L.astype(str)
    return {'LPHVGresults': results, 'P1V': P1V, 'P1L': P1L, 'P2V': P2V, 'P2L': P2L}

def iterfun2(scale:int,len:int):
    """
    生成以scale为步长的索引段,用于PAA
    :param scale:
    :param len:
    :return:
    """
    result=0
    while result<=len:
        if result+scale<=len:
            yield result,result+scale
        else:
             yield result,len
        result+=scale
def MsLPVG(UTS,N,scale):
    """
    对UTS按scalePAA分段均值化,再对新序列进行LPVG
    :param UTS:
    :param N:
    :param scale:
    :return: 合并后的Series
    """
    # datas=[np.mean(UTS[i:j]) for i,j in iterfun2(scale,len(UTS))]
    # df=list(getLPHVG(datas,N))
    return LPHVG([np.mean(UTS[i:j]) for i,j in iterfun2(scale,len(UTS))],N)
def MsLPVG(UTS,N,scale):
    iter1=iterfun2(scale,len(UTS))
    return LPVG([np.mean(UTS[i:j]) for i,j in iter1],N)
# if __name__ == '__main__':
#     np.random.seed(23)
#     sub_data = np.asarray([np.random.random(20), np.random.random(20), np.random.random(20), np.random.random(20),
#                 np.random.random(20)])
#     # print(type(sub_data))
#     VVG(sub_data,5)

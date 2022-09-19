from DaChuang2021.NetAnalysis.degree import *
from DaChuang2021.NetAnalysis.draw_graph import *
import os
def show_data(data,save_to,need_show=True):
    Search_save_to(save_to)
    plt.figure(dpi=300)
    for i in range(data.shape[0]-1):
        plt.subplot(data.shape[0],1,i+1)
        plt.plot(data[i])
        plt.xticks([])


    plt.subplot(data.shape[0],1,data.shape[0])
    plt.plot(data[data.shape[0]-1])
    plt.savefig(save_to+'数据图.png')
    if need_show:
        plt.show()

def read_in_chunks(filePath, chunk_size=1024 * 1024):
    file_object = open(filePath, 'r', encoding='utf-8')
    while True:
        chunk_data = file_object.read(chunk_size)
        if not chunk_data:
            break
        yield chunk_data
def conway(n):
    a=np.array([1,1])
    for i in range(2,n):
        a=np.append(a,a[a[i-1]-1]+a[i-a[i-1]])
    return (a-np.array(range(1,n+1))/2)


if __name__ == '__main__':
    #种子为23
    data=pd.read_csv('testData/BeiJingCleanStandardData.csv',encoding='gbk')
    sub_data=data.iloc[0:,3]
    sub_data=np.asarray(sub_data.loc[:].T)
    print(sub_data.shape)
    # data=pd.read_csv('fig_results/VVG/北京/VVG.csv',encoding='gbk')
    # sub_data=data.loc[:,['VVGi','VVGj']]
    # print(sub_data)
    # G=make_VVGgraph(sub_data)
    # nx.draw(G,)
    # plt.show
    # print(data.iloc[0:5,3:9])
    # print(sub_data.loc[0:10,:].T)
    # np.random.seed(23)
    # sub_data=[np.random.random(20),np.random.random(20),np.random.random(20),np.random.random(20),np.random.random(20)]
    # result=MlLPVG(data,1,5)
    # print(result['P1V'])
    # print(result['P1L'])
    # draw_LPHVG(sub_data[0],1,node_size=20,font_size=5,save_to='fig_results/北京/')
    # make_LPVGgraph(sub_data)
    # save_to='fig_results/VG/北京/'
    # show_data(sub_data,save_to,False)
    # draw_VVG(sub_data,5,save_to,node_size=10,need_show=False,with_labels=True)
    # draw_VG(sub_data,save_to=save_to,node_size=10)
    # save_to='fig_results/HVG/北京/'

    # draw_HVG(sub_data,save_to=save_to,node_size=10)

    # show_data(sub_data,'fig_results/北京/随机/')
    # draw_MlLPHVG(sub_data,Core=5,N=1,node_size=10,save_to='fig_results/北京/随机/MlLPHVG/',need_show=False,need_degree=True)
    # draw_MlLPVG(sub_data,Core=5,N=1,node_size=10,save_to='fig_results/北京/随机/MlLPVG/',need_show=False,need_degree=True)
    # # print(data)
    # data=np.array([0.9, 0.5, 0.4, 0.8, 0.9, 0.5, 0.4, 0.8, 0.9, 0.5, 0.4, 0.8, 0.9])
    # G=draw_LPVG(data[0],1)
    # draw_degree_frequency_distribution(G)
    # print(os.getcwd())
    # print(os.path.exists('testData/electricity.txt'))
    conway(60000)
    plt.plot(range(60000), conway(60000), color="blue", linewidth=2)
    plt.show()
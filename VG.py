# 2.1 最大值分裂算法
"""
不用看,没搞完
"""
# 通过观察发现,在时间序列中,最大值时间点的左边的任意点都看不到其右边的任意点,因此最大值的时间点就将时间序列恰好分裂为相互不可视的2个部分.
# 若先求出时间序列的最大值,则其左右两边就是2个完全独立的时间序列,之后再分别对这2个部分求最大值,于是时间序列被分成独立的4个部分.
# 这样继续分裂下去,直到每个小的时间序列长度为1才停止分裂.具体算法描述如下:
# (ⅰ)初始化网络,将时间序列中的所有数据点添加到网络中;
# (ⅱ)遍历时间序列,找到最大值,将序列分裂为2个部分;
# (ⅲ)分别在2个分序列中找到最大值,将序列继续分裂;
# (ⅳ)重复上述操作,直至分裂后的序列长度为1,算法终止.
# import matplotlib.pyplot as plt
# fig=plt.figure()
# aes=fig.add_subplot(111)
# aes.set(xlim=[0.5,4.5],ylim=[-2,8])
# plt.show()

#思路:制作二叉树,每个节点代表着一段UTS,并记录其最大值的索引.
#以每段的最大值所做处切开UTS(不包含最大值),左半段是左子节点,右半段是右子节点,
#直到片段的长度小于等于3
#还原整个时间序列的遍历是左根右
#对每个叶节点的碎片序列寻找其前后的根节点,形成两边高,中间低的序列,在这短序列里寻找各个是可视对象,再考虑其他根节点能否被可视



#如果采用只记录分隔点的索引的数组能简化程序
#index_list数组中,[1:2]是[0]的子节点,[3:4]是[1]的,[5:6]是[2]的,即2n+1.2n+2是n的子节点,
#填index_list时为保持队形,用-1代表虚节点,如[3,8,1,2,9,1]的indexlist是[4,1,-1],根据索引拆分成[0:1],[1:4],[4:]3个碎片


global INDEX_LIST,INORDER_TRA,CAN_SPLIT,POINT_LIST
def __find_max(UTS:[], need_to_add:int):
    global INDEX_LIST, CAN_SPLIT
    if len(UTS)>3:
        CAN_SPLIT=True
        max_index=UTS.index(max(UTS))#多个最大值时返回第一个索引
        INDEX_LIST.append(need_to_add + max_index)
    else:
        INDEX_LIST.append(-1)
def __iter_yield_index(list:[]):
    """
    如输入[3,6,8],依次生成0,3,3,6,6,8,8,None,可以作为索引形成[0:3],[3:6],[6:8],[8:]
    :param index_list:
    :return:
    """
    yield 0
    for i in range(len(list)):
        if list[i] != -1:
            yield list[i]
            yield list[i]+1
        else:
            j=i
            while list[j]==-1:
                j=j-1
            if j>=0:
                yield list[j]
                yield list[j]+1
            else:
                yield 0
                yield 0
    yield None

def __do_inorder_traversal(n:int):
    global INORDER_TRA
    # 左中右序遍历输出索引列表
    if n<len(INDEX_LIST):
        __do_inorder_traversal(2 * n + 1)
        INORDER_TRA.append(INDEX_LIST[n])
        __do_inorder_traversal(2 * n + 2)

def __get_index(data:[]):
    """
    反复调用FindMax来切割数据
    :return:
    """
    global INDEX_LIST, CAN_SPLIT
    __do_inorder_traversal(0)
    CAN_SPLIT=False #假定这轮是最后一轮切割(即添入index_list全是-1),只有在调用FindMax时有一次切割成功就被推翻
    if len(INDEX_LIST)==1:
        __find_max(data[0:INDEX_LIST[0]], 0)
        __find_max(data[(INDEX_LIST[0] + 1):], (INDEX_LIST[0] + 1))
    else:
        Order=__iter_yield_index(INORDER_TRA)
        while True:
            # 以下还要调整,当出现[-1,0,0,...]\[-1,....,len(data)-1,None]....竟然全能运行结果也没问题,牛鼻
            try:
                temp1=next(Order)
                temp2=next(Order)
                __find_max(data[temp1:temp2], temp1)
                # FindMax(data[temp:next(Order)],temp)
            except StopIteration:
                return


class __point_list(object):
    def __init__(self,data:[],index_list,inorder):
        self.data=data
        self.visual_list=[list() for i in range(len(data))]# 创建一个[[],[],[],...,[],]的二维数组
        self.max_index=index_list
        inorder=list(self.__iter_yield_short_ts(inorder))
        inorder=[inorder[i:i + 2] for i in range(0, len(inorder), 2)]
        for item in inorder:
            self.__search_visual_point_inside(item[0],item[1])
            self.__search_visual_point_outside(item[0], item[1])


    def __iter_yield_short_ts(self,inorder:[]):
        yield 0
        for i in inorder:
            if i!=-1:
                yield i
                yield i+1
        yield None

    def __search_visual_point_inside(self, lower, upper):
        # 根据中序遍历的index_list出的每个碎片的索引上下界,在每个碎片里寻找可视对象,并填入mypoint_list,可并行执行,待优化
        if upper is None:
            return
        elif upper-lower<=2:
            return
        elif upper-lower==3:
            if self.data[lower + 1]-self.data[lower]<(self.data[lower + 2] - self.data[lower])/2:
                self.visual_list[lower].append(lower + 2)
                self.visual_list[lower + 2].append(lower)
        elif upper-lower==4:
            if self.data[lower + 1]-self.data[lower]<(self.data[lower + 2] - self.data[lower])/2:
                self.visual_list[lower].append(lower + 2)
                self.visual_list[lower + 2].append(lower)
            if self.data[lower + 1]-self.data[lower]<(self.data[lower + 3] - self.data[lower])/3:
                self.visual_list[lower].append(lower + 3)
                self.visual_list[lower + 3].append(lower)
            if self.data[lower + 2]-self.data[lower + 1]<(self.data[lower + 3] - self.data[lower])/2:
                self.visual_list[lower + 1].append(lower + 3)
                self.visual_list[lower + 3].append(lower + 1)
    def __search_visual_point_outside(self, lower, upper):
        if upper is not None:
            index=self.max_index.index(upper)
        else:
            index=self.max_index.index(lower-1)
            upper=len(self.data)
        to_try_list=[]
        while index>0 & index<len(self.max_index):
            index=(index-1)//2
            to_try_list.append(self.max_index[index])

        for i in range(lower, upper):
            k1 = self.data[i+1]-self.data[i]
            for j in to_try_list:
                k2=(self.data[j]-self.data[i])/(j-i)
                if k2>k1:
                    self.visual_list[i].append(j)
                    self.visual_list[j].append(i)
    def sort_visual_list(self):
        for item in self.visual_list:
            item.sort()
    def print_visual_list(self):
        return self.visual_list

def main(data:[]):
    global INDEX_LIST,CAN_SPLIT,INORDER_TRA
    INDEX_LIST=[]
    INORDER_TRA=[]
    CAN_SPLIT=False
    __find_max(data, 0)
    while CAN_SPLIT:
        __get_index(data)
        INORDER_TRA=[]
    else:
        __do_inorder_traversal(0)
    data=__point_list(data,INDEX_LIST,INORDER_TRA)
    print(data.visual_list)

if __name__=="__main__":
    UTS=[2,4,3,5,6,8,1,5,2,7,4,6,12,7,9,6,5,4,3,2,5]
    INDEX_LIST=[12,5,14]
    main(UTS)

    

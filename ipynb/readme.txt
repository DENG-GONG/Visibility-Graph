此处给出了10多种VG方法的实现代码。

在各个VG方法的文件中，我们基本给出以下三种类型的函数：
①get_Point函数：输入序列，输出网络中需要连接的两点，基于该函数可构建出各可视图网络。
②bar函数：输入序列，该函数会调用get_Point函数，绘制出对应的条形图，便于更直观的理解可视图网络的形成。
③net函数：输入序列，该函数会调用get_Point函数，绘制形成的网络，便于直观的理解。

此外，在fast_methods文件中，我们对get_Point函数的效率进行了提升，
并提供了一些邻接矩阵、度分布图的绘制方法以及网络属性计算方法。

注：虽然我们对函数进行了优化，但由于能力有限，在实际使用时，仍存在效率较低的问题。
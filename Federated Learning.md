# 联邦学习论文

## 1、FedAvg([Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629))

### 期刊： *AISTATS 2017*

### 针对场景、问题：

数据的敏感性和隐私性问题越来越被重视。同时，大规模数据使用也存在一些问题：比如数据标签的不平衡、数据在不同储存处之间是非IID的。

如果将所有的数据都进行通信，传到某一个节点进行大规模训练，不仅不利于保护数据的隐私，（因为一定存在某一个节点掌握很多的数据）并且在通信过程中，会消耗很多的带宽，带来巨大的通信成本。

### 本文主要方法：

![截屏2022-10-27 下午6.18.43](https://github.com/HyperJupyterTale/Stay-Hungry/blob/main/images/截屏2022-10-27%20下午6.18.43.png)

本质上就是将每个节点训练的模型的参数，汇总到server端进行均值聚合。

### 此方法相较于其他方法的优越性：

#### 相比于传统方法：

1、解决隐私问题：本地模型的数据不会上传的服务器。client会帮助server维护当前全局模型的更新。全局模型是储存在server端的。这个全局模型会被进行通信，下载到client端。

2、解决安全问题：将训练模型和原始数据分开，client训练的模型被攻击仅限制在边缘设备，不会对总的模型产生影响。

3、分布式计算可以更好的利用边缘计算资源。

#### 相比于FedSGD：

1、FedSGD：同样只是选择了一部分client进行每轮的通信，不是所有的client都会参与。FedSGD每一轮通信会默认选择本地所有的数据为一个batch，训练一次之后上传至server，server进行平均融合。因此需要很多轮才能收敛。

2、FedAvg：FedAvg会先在本地进行多轮次、多个批量训练更新本地权重。每次使用小批量进行多次训练。再将训练的权重w传至server端进行平均融合。思想是增加并行/增加每个client的计算量。

### 创新点：

1、提出了一种基于移动设备的分布式的研究方向。

> The identification of the problem of training on decentralized data from mobile devices as an important research direction.

2、提出了一种直接有效的算法来应用于分布式移动设备。

> The selection of a straightforward and practical algorithm that can be applied to this setting.

3、对提出的方法进行广泛评估。

> An extensive empirical evaluation of the proposed approach.

前三点是文章中写的

4、提出了FedAvg的适用场景，以及一些面临的问题。

### 是否开源：

[是](https://github.com/roxanneluo/Federated-Learning)

### 数据集：

CIFAR，MNIST，莎士比亚戏剧集

### 注意事项：

1、FedAvg算法是说固定batch在本地更新w权重之后，上传到server端。batch可能不是Pk的数量，很可能小于Pk。一次更新确实有可能不用所有的数据，但是上传的数据一定是遍历整个数据集的。

2、每次在本地训练的时候，都要保证初始化权重参数必须是一样的。这也是为什么需要在多轮迭代的时候每次服务器再把训练好的权重下载到每个client上。不然的话就相当于非IID的分布，效果会变差。

3、这个模型可行的前提是：client信任server。

4、文章中提到：C并不是越大越好。可以作为探索的点思考。

### 个人理解：

1、有了数据之后可以干很多不同的事情，我们不知道拥有这部分数据的人会拿来做什么。但是将其训练成为模型之后，数据的被模型提取特征局限于某部分，更新这部分参数并不会对用户的数据造成严重的威胁。模型的参数相比于原数据而言，可以获取到的信息会少很多。本文所构想的，是一种“数据就是用来更新模型的，如果我能做到更新全局模型，其实可以不储存数据。”的理念。

> Since these updates are specific to improving the current model, there is no reason to store them once they have been applied.

2、如果训练样本越不均衡，batch size越大对分类的结果影响也越大。因为大量的同类型的数据会把梯度带偏，使得损失函数都在优化某些大量数据的梯度。batch size过大（比如一整个数据集）会占用很多资源，使更新变缓慢，而且容易陷入局部最优。batch size过小（比如一个数据）不容易完全收敛，但是能够避免陷入局部最优，因为数据本身单个采样会有随机性，从某种程度上加入了随机性也会避免过拟合。一般来说大的batch size伴随着更大的学习率，不然训练的时间就会大大加长。因此，最好的办法是选取一个批量进行更新权重w，既不过大，也不过小。

3、思考FedAvg与dropout正则化的关系。



## 2、FedProx([Federated Optimization for Heterogeneous Networks](https://openreview.net/pdf?id=SkgwE5Ss3N))

### 期刊：*MLSys 2020b*

### 针对场景、问题：

每个client的数据类型很有可能不同，从而导致了数据异构的问题。数据异构会导致在本地训练的模型只能学习到异构数据的特征，往往具有明显的倾斜性。这些模型的权重被server端汇总之后，再平均相加的效果会大打折扣。

### 本文主要方法：

![截屏2022-10-30 下午2.56.32](https://github.com/HyperJupyterTale/Stay-Hungry/raw/main/images/截屏2022-10-30%20下午2.56.32.png)

本质上就是对本地的损失函数增加了一项正则化。通过约束本地参数w尽量不偏离本次server端的参数太多，从而实现对数据异构的缓解。

### 此方法相较于其他方法的优越性：

#### FedAvg：

收敛条件：(1)数据要么在设备间共享，要么以IID方式分布，或者(2)所有设备在每轮通信中都是活跃的。

> (1) the data is either shared across devices or distributed in an IID (identically and independently distributed) manner, or (2) all devices are active at each communication round

#### FedProx：

通过正则化项的约束，往往能够更快的达到收敛。同时对数据异构有所缓解，对异构数据的模型效果比FedAvg好。

### 创新点：

1、我们提出了一个用于异构网络的联合优化框架，即FedProx，它包含了FedAvg。为了描述FedProx的收敛行为，我们在网络中引用了一个设备不相似性的假设。最后，我们证明了我们的理论假设反映了经验性能，并且当数据在设备间异质化时，FedProx可以比FedAvg提高收敛的稳健性和稳定性。

> We propose a federated optimization framework for heterogeneous networks, FedProx, which encompasses FedAvg. In order to characterize the convergence behavior of FedProx, we invoke a device dissimilarity assumption in the network. Under this assumption, we provide the first convergence guarantees for FedProx. Finally, we demonstrate that our theoretical assumptions reflect empirical performance, and that FedProx can improve the robustness and stability of convergence over FedAvg when data is heterogeneous across devices.

以上是文章中说的。

2、提出了一个可以描述数据异构的评价准则。（Definition2，B-local dissimilarity）

3、提出了一种思想，可以不直接把每个client的参数上传，而是经过一定的约束再上传。虽然本质上还是通过优化损失函数，改变参数的值，然后直接上传，但是以文中逻辑，就不是直接上传原始训练的本地模型参数，而是可以和客户端传回的参数值建立联系，达到某种目的。

### 是否开源：

[是](https://github.com/litian96/FedProx)

### 数据集：

FEMNIST，莎士比亚戏剧集

### 注意事项：

1、本文中有大量的数学证明收敛性，包括附录在内。这些东西直接跳过了，说结论：选取合适的u可以加速模型收敛（相较于FedAvg）

2、FedProx可以向下兼容FedAvg，当且仅当u=0时。

### 个人理解：

1、其实本地参数的训练不止只能从本地数据集中来，还有别的信息可以使用。比如每轮server端的参数传回。

2、[正则化项可以防止过拟合，也可以加速收敛。](https://blog.csdn.net/yinyu19950811/article/details/61922893) 本文使用了正则化来解决异构和收敛问题，本质上是有对应关系的。

## 3、 FedPer([Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818.pdf))

### 期刊：无(arxiv)

### 针对场景、问题：

本文主要针对数据异构提出，为了解决server端模型不能处理本地用户的个性化问题。比如相同的数据在不同的用户客户端上面接收不同的标签。

> Hence, same input data can receive different labels from different users implying that the personalized models must differ across users to be able to predict different labels for similar test data.

同时本文的模型能够解决在小数据量上面训练模型的问题。

### 本文主要方法：

![截屏2022-10-30 下午8.34.41](https://github.com/HyperJupyterTale/Stay-Hungry/blob/main/images/截屏2022-10-30%20下午8.34.41.png)

WP是个性化参数，放在WB层的后面。更新只更新WB参数，不更新本地的WP参数，这样可以保持个性化的预测。

### 此方法相较于其他方法的优越性：

相比于FedAvg而言，提升了针对于本地用户的个性化需求。并且在小数据量上有着不错的表现。在训练阶段，能够有更高的准确率和更快的收敛。

### 创新点：

我们建议通过将深度学习模型看作基础+个性化层来捕获联邦学习中的个性化方面。我们的训练算法包括通过联邦平均(或其变体)训练的基本层和仅从具有随机梯度下降(或其变体)的局部数据训练的个性化层。我们证明了不受联邦平均(FedAvg)过程影响的个性化层可以帮助对抗统计异质性的不良影响。

> We propose to capture personalization aspects in federated learning by viewing deep learning models as base + personalization layers as illustrated. Our training algorithm comprises of the base layers being trained by federated averaging (or some variant thereof) and personalization layers being trained only from local data with stochastic gradient descent (or some variant thereof). We demonstrate that the personalization layers that are free from the federated averaging (FedAvg) procedure can help combat the ill-effects of statistical heterogeneity.

主要是提供了本地不是所有的参数都需要上传，完全可以进上传部分参数，将剩下的一部分留作“个性化”的需求，不进行同步。

### 是否开源：

否

### 数据集：

CIFAR-10/CIFAR-100，FLICKR-AES

### 注意事项：

1、文章中提到一点：在医疗领域中，可以存在假设：可以将全局训练步骤与后续的本地个性化步骤分离开来。（Chen, Y., Wang, J., Yu, C., Gao, W., and Qin, X. (2019). FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare. ArXiv, abs/1907.09173.）（Vepakomma, P., Gupta, O., Swedish, T., and Raskar, R. (2018). Split learning for health: Distributed deep learning without sharing raw patient data. ArXiv, abs/1812.00564.）

2、作者做出了四个宽泛性假设：(a)任何客户端的数据集不会在全局聚合中发生变化，(b)batch size和epoch在客户端和全局聚合中是不变的，(c)每个客户端使用SGD在全局聚合之间更新(WB, WPj)， (d)在整个训练过程中所有N个用户设备都是可以提供更新的。

> a) the dataset at any client doesn’t change across global aggregations, (b) the batch size b and the #epochs e are invariant across clients and across global aggregations, (c) each client uses SGD to update (WB, WPj ) between global aggregations, and (d) all N user devices are active throughout the training process.

### 个人理解：

1、其实层与层之间的参数不是都需要上传，也不是都需要更新的，进行更细规模的区分，从而选择性的上传、更新，可以从某种程度上保存个性化的参数，也能够获取全局模型的参数。获取的全局模型的参数，可以聚合多个模型的参数，从而在小的数据量的情况下，仍然可以得到较好的训练结果。

2、值得思考的一点是：为什么选择了个性化的层在基础层的后面而不是前面呢？在神经网络中，偏后面的层是更general的细节，偏前面的层是更spesific的细节。学习个性化的不应该是spesific的细节会有更多的不一样吗？难道是因为最后和分类层相邻，和最后类别判定强相关，所以才选择了后面的层？如果颠倒一下会怎么样？

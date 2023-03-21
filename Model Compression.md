# 模型压缩论文

## 1、[**Distilling the Knowledge in a Neural Network**](https://arxiv.org/pdf/1503.02531.pdf)

### 期刊：*NIPS 2015*

### 针对场景、问题：

最开始提出是因为很多最新的模型是采用类似voting机制来提高模型性能的。在同一个数据集上训练多个模型，然后取平均来提高性能。但是这样是非常浪费算力和性能的。需要一个模型压缩的算法将这些大的复杂的模型进行融合，减少参数量。

### 本文主要方法：

#### 由一个大模型训练单个小模型

1、训练一个复杂的teacher模型。可以有很多层、很多参数。

2、创建一个student模型，其参数量要远远小于teacher模型。但是student模型的损失函数由两部分构成，一部分是正确的label损失，另一部分是与teacher模型logits数值的MSE。（后面可以使用KL散度来进行替代，因为KL散度描述了两者概率分布的差异情况）因为在原文中，认为teacher模型中概率小的预测值也包含了某些信息，这些信息对于模型的预测至关重要。

3、选取合适的温度值T来进行训练。

![截屏2022-11-19 上午8.42.49](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-19 上午8.42.49.png)

#### 由一个大模型训练多个专家模型

我们首先训练一个通用模型，然后使用混淆矩阵来定义专家被训练的子集。一旦这些子集被定义，就可以完全独立地训练专家。在测试时，我们可以使用通用模型的预测来决定哪些专家是相关的，并且只需要运行这些专家模型。

> We first train a generalist model and then use the confusion matrix to define the subsets that the specialists are trained on. Once these subsets have been defined the specialists can be trained entirely independently. At test time we can use the predictions from the generalist model to decide which specialists are relevant and only these specialists need to be run.

### 此方法相较于其他方法的优越性：

#### 和只使用logits作为损失函数的模型对比：

这种模型相当于只考虑了soft label，只是在输入softmax之前的logits值，并且没有经过平滑处理，即除温度的操作。损失曲面本身没有经过除温度之后的平滑，并且没有了hard label的矫正。

后续作者说明，如果有一部分添加的新内容，即true label的微调，结果会更好。

> We have found that using the original training set works well, especially if we add a small term to the objective function that encourages the small model to predict the true targets as well as matching the soft targets provided by the cumbersome model.

#### 和只使用hard label的模型对比：

如果只使用hard target 效果并不好。

> We have recently become aware of related work on learning a small acoustic model by matching the class probabilities of an already trained larger model [8]. However, they do the distillation at a temperature of 1 using a large unlabeled dataset and their best distilled model only reduces the error rate of the small model by 28% of the gap between the error rates of the large and small models when they are both trained with hard labels.

并且只使用hard target容易过拟合。加上soft target之后能够大大的缓解。

|       System & training set       | Train Frame Accuracy | Test Frame Accuracy |
| :-------------------------------: | -------------------- | :------------------ |
|  Baseline (100% of training set)  | 63.4%                | 58.9%               |
|   Baseline (3% of training set)   | 67.3%                | 44.5%               |
| Soft Targets (3% of training set) | 65.4%                | 57.0%               |

#### 和门控选择专家模型对比：

本文提出的先用KNN进行分类，再通过对类别进行学习的方法具有并行性，而且如果将分类目标设置为全部而非进入trash，往往能够学习到更多的知识。当然如果是具有明显分类的大分类集中的子集，本文提出的混合学习方法就比较有劣势。

> Using the discriminative performance of the experts to determine the learned assignments is much better than simply clustering the input vectors and assigning an expert to each cluster, but it makes the training hard to parallelize: First, the weighted training set for each expert keeps changing in a way that depends on all the other experts and second, the gating network needs to compare the performance of different experts on the same example to know how to revise its assignment probabilities. These difficulties have meant that mixtures of experts are rarely used in the regime where they might be most beneficial: tasks with huge datasets that contain distinctly different subsets.

### 创新点：

1、提出了之前的“谬误”，认为知识就只是模型的参数。如果一定要保持这种参数的不变，实际上是很难进行有效压缩的。但是如果将知识看作是抽象的映射关系，则可以打破这种禁锢。顺理成章的提出错误分类的概率也蕴藏着一些信息，比如文章中举的例子。一个概率比另一个概率大就说明了一些“知识”。并将其命名为 “soft targets”，这也是我们的模型需要学习的目标之一。

2、创造性地引入了温度的概念，让soft target可以更加的平滑，有利于训练的收敛和精度。

3、在MNIST的实验上，证明了确实可以在没有出现的类别上有所提升，能够学习到soft targets。

4、文章中提到了一种可以将一个大的模型分解成为多个专家模型的方法。可以将某个专家擅长的领域中的类进行分类，将其它的类归为trash。具体的分类可以采用KNN算法实现。也可以使用所有的类来训练专家类，理论上更有优势。

> If we allow specialists to have a full softmax over all classes, there may be a much better way to prevent them overfitting than using early stopping. A specialist is trained on data that is highly enriched in its special classes. This means that the effective size of its training set is much smaller and it has a strong tendency to overfit on its special classes. This problem cannot be solved by making the specialist a lot smaller because then we lose the very helpful transfer effects we get from modeling all of the non-specialist classes.

### 是否开源：

[否](https://github.com/tangchen2/Model-Compression)

### 数据集：

MNIST，CIFAR10，ASR language system（自己做的）

### 注意事项：

1、本文提到“软标签”的训练需要乘T^2的参数，因为要平衡软硬标签的权重。这是根据数学公式推导而来。

2、文章中提到的方法是可以并行进行训练的。因为不是门控选择对应的专家模型来进行推断，而是使用KNN的方法。

### 个人理解：

1、定性的解释了为什么大模型变小模型需要更少的数据来训练和更高的学习率。

> When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate.

为什么相对均匀的概率会有更小的梯度方差呢？这个从数学公式推导可以解决，每一个预测的概率都接近的话，确实会减少梯度的波动性。为什么可以减少数据集的数量？因为梯度的方差减小了，梯度更加平滑，减少了梯度下降过程中的崎岖。局部最优更接近于全局最优。为什么可以提高学习率来训练？同样的，因为损失函数梯度曲面更加平滑。

2、为什么大模型到小模型不只采用softmax来进行筛选呢？因为大的概率越来越大，小的越来越小，会让那些小概率互相之间的关系被遮盖。不符合最开始作者提出的小概率和小概率之间也存在某些关系。因此，更倾向于使用logits数值来进行计算。（logits数值是指在softmax层之前那一层输出的数值）

3、文章从某种程度上说明了为什么直接用小模型效果不好：不能得到原始模型的“泛化性”信息。

> It is generally accepted that the objective function used for training should reflect the true objective of the user as closely as possible. Despite this, models are usually trained to optimize performance on the training data when the real objective is to generalize well to new data. It would clearly be better to train models to generalize well, but this requires information about the correct way to generalize and this information is not normally available.

文章中的4.1Results中的第一段也进行了实验的补充，在每一个单独小专家模型上让它去看更多类型的数据，结果并没有更好。

> We have explored adding diversity to the models by varying the sets of data that each model sees, but we found this to not significantly change our results, so we opted for the simpler approach.

## 2、[Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/pdf/1405.3866.pdf)

### 期刊：BMVC 2014

### 针对场景、问题：

在图像处理过程中，往往使用卷积神经网络。然而，为了比较好的性能，网络本身可能会包含很多的参数量以及层数。这些参数有很多是冗余的。

本文主要解决减少卷积神经网络参数量、加速运算、减少计算时间的问题。同时该方法也一定程度上解决了硬件的问题，可以在GPU/CPU上匹配适用。

### 本文主要方法：

1、理论分析了通过低阶近似矩阵来逼近原始卷积层可以减少参数量。

2、提出了两种低阶近似的方法。

原始卷积层方法：

![截屏2022-11-14 上午10.32.33](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-14 上午10.32.33.png)

第一种方法是先通过M组两个rank=1的矩阵逼近原始的d * d的2D卷积核，得到M个通道的具有原始二维卷积操作同等大小的feature map，其中M<N，之后再通过N个1 * 1 的卷积核还原原始通道维度：

![截屏2022-11-14 上午10.32.46](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-14 上午10.32.46.png)

第二种方法是先通过K个rank=1的向量，得到中间feature map，再通过N组d * 1 * K的矩形卷积，近似还原得到原始feature map：

![截屏2022-11-14 上午10.33.32](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-14 上午10.33.32.png)

这两种方法的区别：作者认为，第一种方法只能减少输出的N通道的冗余。但是第二种方法可以同时减少输入和输出的冗余。

> Scheme 1 focuses on approximating 2D filters. As a consequence, each input channel zc is approximated by a particular basis of 2D separable filters. Redundancy among feature channels is exploited, but only in the sense of the N output channels. In contrast, Scheme 2 is designed to take advantage of both input and output redundancies by considering 3D filters throughout. The idea is simple: each convolutional layer is factored as a sequence of two regular convolutional layers but with rectangular (in the spatial domain) filters.

3、使用作者提出的损失函数对整个过程优化。作者在文中同样提出了两种损失函数。第一种方法旨在通过最小化滤波器重建误差直接重建原始滤波器。第二种方法通过最小化卷积层输出的重建误差，间接地接近卷积层。

> The first method aims to reconstruct the original filters directly by minimizing filter reconstruction error. The second method approximates the convolutional layer indirectly, by minimizing reconstruction error of the output of the layer.

第一种Filter Reconstruction Optimization描述的是和原始滤波器的参数距离的差值，文中称之为重建损失(reconstruction error)。需要注意的是，2种提出的两种方法分别具有不同的损失函数。

第一种方法的损失函数：

![截屏2022-11-14 上午10.58.42](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-14 上午10.58.42.png)

第二种方法的损失函数：

![截屏2022-11-14 上午10.59.02](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-14 上午10.59.02.png)

这两种方式损失函数不一样的原因目前也不太懂，可能和公式流程推导不一样有关。

第二种Data Reconstruction Optimization描述的是从第一层到第l层所有的误差。这个损失函数的提出是针对于Filter Reconstruction Optimization的目标是最小化和原始滤波器的差异最小化。但是这样做可能并不有助于模型最终分类等任务的提升。

> The problem with optimizing the separable basis through minimizing original filter reconstruction error is that this does not necessarily give the most optimized basis set for the end CNN prediction performance. As an alternative, one can optimize a scheme’s separable basis by aiming to reconstruct the outputs of the original convolutional layer given training data.

![截屏2022-11-14 上午11.06.48](/Users/mac/Library/Application Support/typora-user-images/截屏2022-11-14 上午11.06.48.png)

这种方法在优化近似方案方面有两个主要优势。首先，近似是以训练数据的流形为条件的——在训练数据背景下不相关或多余的原始滤波器尺寸将被最小化数据重建误差所忽略，但仍会被最小化滤波器重建误差所惩罚，因此会无用地占用模型容量。其次，堆叠的近似层可以通过将数据通过近似网送至第l层而不是原始网送至第l层来学习，以纳入前几层的近似误差。这还意味着所有的近似层都可以用反向传播法进行联合优化。

> There are two main advantages of this method for optimization of the approximation schemes. The first is that the approximation is conditioned on the manifold of the training data – original filter dimensions that are not relevant or redundant in the context of the training data will by ignored by minimizing data reconstruction error, but will still be penalised by minimizing filter reconstruction error and therefore uselessly using up model capacity. Secondly, stacks of approximated layers can be learnt to incorporate the approximation error of previous layers by feeding the data through the approximated net up to layer l rather than the original net up to layer l . This additionally means that all the approximation layers could be optimized jointly with back-propagation.



### 此方法相较于其他方法的优越性：

#### 低秩分解与FFT对比

FFT也是加速单个卷积，与本文提出的低秩分解类似。然而，本方法拥有如下优势：比调整FFT实现更容易，特别是在GPU上；低秩分解不需要将特征图填充到一个特殊的大小，比如数量限定在2的幂；低秩分解的内存效率要高得多；而且，低秩分解对小图像和过滤器的大小也有很好的加速效果，而FFT加速往往对大过滤器更好，因为计算FFT时产生开销。

> Note that the FFT could be used as an alternative speedup method to accelerate individual convolutions in combination with our low-rank cross-channel decomposition scheme. However, separable convolutions have several practical advantages: they are significantly easier to implement than a well tuned FFT implementation, particularly on GPUs; they do not require feature maps to be padded to a special size, such as a power of two as in [21]; they are far more memory efficient; and, they yield a good speedup for small image and filter sizes too (which can be common in CNNs), whilst FFT acceleration tends to be better for large filters due to the overheads incurred in computing the FFTs.

#### 低秩分解第二种损失函数与借助未优化的近似层对比

一个明显的替代性优化策略是用未优化的近似层取代原始卷积层，并通过反向传播近似CNN的分类误差来训练这些层。然而，这实际上并不比在实践中做L2数据重建优化有更好的分类精度，在全网内优化可分离基础会导致训练数据的过拟合，而试图通过正则化方法（如dropout）来最小化这种过拟合，会导致欠拟合，这很可能是由于我们已经在试图大量地近似我们的原始过滤器。

> An obvious alternative optimization strategy would be to replace the original convolutional layers with the un-optimized approximation layers and train just those layers by backpropagating the classification error of the approximated CNN. However, this does not actually result in better classification accuracy than doing L2 data reconstruction optimization in practice, optimizing the separable basis within the full network leads to overfitting of the training data, and attempts to minimize this overfitting through regularization methods like dropout lead to under-fitting, most likely due to the fact that we are already trying to heavily approximate our original filters. 

### 创新点：

1、引入了多种低秩分解的方法。能够将一个多通道较大的卷积层操作近似分解成多个比较小的向量卷积共同作用的形式，大幅度减少了计算量。

2、提出了多种适合于低秩分解的损失函数。

3、设计实验验证了不同方式的低秩分解以及运用不同损失函数的理论值和真实值，并且分析了可能出现这种问题的原因。

### 是否开源：

否

### 数据集：

ICDAR2003，ICDAR2005，ICDAR2011，

http://algoval.essex.ac.uk/icdar/datasets.html

http://www.iapr-tc11.org/mediawiki/index.php/kaist_scene_text_database

T. de Campos, B. R. Babu, and M. Varma. Character recognition in natural images. 2009.

T. Wang, D. J. Wu, A. Coates, and A. Y. Ng. End-to-end text recognition with convolutional neural networks. In Pattern Recognition (ICPR), 2012 21st International Conference on, pages 3304–3308. IEEE, 2012.

### 注意事项：

1、低秩分解的卷积层，不用于第一层和最后一层。第一层作用于原始的图像像素，并且针对于最基础的特征提取，文中也说明做了一些相关的实验证明分解第一层并不是一个很好的选择。最后一层是因为最后一层的卷积大多是1 * 1的卷积核，没有分解的必要。

> For the CNN presented, we only approximate layers Conv2 and Conv3. This is because layer Conv4 has a 1 × 1 filter size and so would not benefit much from our speedup schemes. We also don’t approximate Conv1 due to the fact that it acts on raw pixels from natural images – the filters in Conv1 are very different to those found in the rest of the network and experimentally we found that they cannot be approximated well by separable filters .

2、理论和实际的差异可能很大，这是因为框架本身的加速方式以及硬件所共同决定的。因此实际加速过程中，还需要对这部分有一定的理解。

### 个人理解：

低秩近似的过程其实可以看作是外积的过程，很多多模态方向的也有运用外积来寻找不同维度之间的某些关系的。具有一定的共通性。

低秩分解本身还是希望能够减少计算量和参数量。而卷积的过程，本质上就是线性代数的运算。因此，低秩分解其实是一个非常数学的过程。个人感觉只在这种有卷积，并且可以使用数学方式减少计算的时候才有应用的可能。还是具有一定限制条件的。

并且在操作过程中，需要考虑硬件以及框架的支持程度。当然，不止是这一种方法，剪枝等模型压缩的方法也是需要的。

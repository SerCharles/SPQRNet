1.使用1个gt对应12个partial的数据集，训练PCN达到温欣的baseline（多种数据一起训练）

结果：



可视化结果：



2.用之前PCN的baseline搭建网络，使用triplet loss，占比和重建在一个数量级上，调试三个重建loss（anchor，positive，negative）的去留

结果：



可视化结果：



3.用之前的loss去留，比较triplet，cosine，contrastive以及其不同的超参数，找最好的loss和超参数

结果：



可视化结果：



4.改变triplet类loss的占比，比较分析loss的作用

结果：



可视化结果：

5.用之前的超参数，修改训练方式，比较结果（端到端，同时训练triplet类和decoder类；先不用triplet进行训练，然后迁移学习初始化，triplet finetune；先训练triplet和decoder，之后迁移学习初始化只训练decoder。。。。）

结果：



可视化结果：



6.参考siamese网络的encoder-decoder结构

 https://arxiv.org/abs/2001.09650， The Whole Is Greater Than the Sum of Its Nonrigid Parts
---
typora-root-url: ..\result\pictures
---



#### 1.使用1个gt对应12个partial的数据集，训练PCN达到温欣的baseline（多种数据一起训练）

shapenet数据集+topnet baseline：抄源代码的超参数，检查后完全一致（除了EMD和学习率衰减没用）

**baseline:18.22   我：19.41**

**在chair类上单独训练，我自己的没用任何loss的网络还比PCN原版略好**

![pcns](/pcns.png)

#### 2.用之前PCN的baseline搭建网络，使用triplet loss，占比和重建在一个数量级上，调试三个重建loss（anchor，positive，negative）的去留

几种配比大致相当

![rec_weights](/rec_weights.png)

#### 3.用之前的loss去留，比较triplet，cosine，contrastive loss

triplet loss完全是捣乱

![triplet_weights](/triplet_weights.png)

cosine loss和带norm的 cosine loss影响不大

![cosine_weights](/cosine_weights.png)

Contrastive loss：todo，这周各种事情和考试，9.3再说

#### 4.对于先前最优的loss进行调参

triplet是负优化效果就8调了，调cosine的margin即可

![cosine_margins](/cosine_margins.png)

![norm_margins](/norm_margins.png)

cosine loss大概都差不多



Contrastive loss：todo，这周各种事情和考试，9.3再说

#### 5.用之前的超参数，修改训练方式，比较结果（端到端，同时训练triplet类和decoder类；先不用triplet进行训练，然后迁移学习初始化，triplet finetune；先训练triplet和decoder，之后迁移学习初始化只训练decoder。。。。）

todo，这周各种事情和考试，9.3再说

#### 



#### 6.参考siamese网络的encoder-decoder结构

 https://arxiv.org/abs/2001.09650， The Whole Is Greater Than the Sum of Its Nonrigid Parts

todo，这周各种事情和考试，9.3再说
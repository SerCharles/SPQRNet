

# Weakly learning（Semi Supervised） point cloud completion with joint feature（domain adaption） projection and xxx

## Note

### 1.**Motivation**.

实际采样得到的点云都是不均匀（比如ridar采集，正对着等部分密集，而背对着等部分很稀疏），不完整（比如scannet的桌椅有严重的遮挡问题）的，因此需要补全。而虽然学术界已经有Shapenet这种完整等点云数据集，但是大多数情况下获得等点云（快手数据集，scannet，kitti等，用ridar扫描等）都是不完整，无标注等，获取有标数据难度大（CAD软件画）。如何利用少数有标注数据集来改进大多数无标注点云等补全，就很重要。

### 2.**Related work**.

- PCN（目前主要用的），GRNet，MSN，TopNet等有监督补全的work
- 2017ACMMM - Adversarial Cross-Modal Retrieval 跨模态融合方法---之后考虑采用不同的融合方法
- PCL2PCL等利用GAN来无监督进行点云操作的work
- weakly learning。。。
- few shot

### 3.**Problem**. 

- 综合利用有标数据（shapenet）和无标数据（scannet）来提高无标数据的点云补全效果
- 如果使用有监督方法（shapenet训练scannet测试），因为两个数据集有GAP，效果不一定好
- 如果使用自监督学习方法（天阳那个），只能在shapenet完整的点云上训练，在scannet上无法这样做
- 如果使用基于cycleGAN的无监督方法（温欣那个），综合利用了shapenet和scannet，但是没有充分利用shapenet的有标数据

### 4.**Idea.**

- 是否可以采用跨模态融合的思路，用feature projection来将有标数据集和无标数据集的特征映射到一个分布内，然后用分类（暂时只用这个），重建等来训练这个映射网络？
- 是否在decoder部分可以采用PCL那种GAN，来改进decoder效果？
- 暂时不做）是否除了PCN，PCL，DNN之外，在encoder，decoder，融合，多任务方面，还有其他的改进方法？参考最新的论文？采用自己的方法？
- 新任务与loss：triplet loss
- 辨别模式：用新的辨别方式：real？shapenet？

### 5.**Keywords**. 

### 6.**Title**. 

### 7.**Application**. 

利用少数有标注数据集来改进大多数无标注点云等补全，比如补全kitti，scannet，快手数据集

### 8.**Benchmark**. 

用PCN补全shapenet，以及shapenet训练的PCN直接用于scannet

（暂时不用）用最好的有监督算法替代PCN的情况，最好的基于无监督GAN的方法的情况

### 9.**Metric**.

- 这个补全shapenet的CD，EMD值
- 补全的scannet和原有scannet的CD,EMD值
- 补全scannet的可视化

 

### 10.**Limitation**.

### 11.**Innovation**.

### 12.**Contribution**.

### 13.**Methods**. 



## Reference

 

## Scheme

1. 完成实验计划，绘制网络图
2. 看2017ACMMM - Adversarial Cross-Modal Retrieval，PCL的论文，源代码，实现网络
3. 在shapenet训练，scannet测试，至少用椅子一类数据，得到对应的两组loss和可视化结果
4. 用PCN在shapenet训练，scannet测试，得到结果进行对比
5. 整理实验结果等，完成小学期答辩

----------------------------保研分割线------------------------------

1. encoder部分，尝试用最新的GRNet，MSN等方法，或者自己设计网络
2. decoder部分，尝试用更新的基于CYCLEGAN的无监督方法
3. concat部分，尝试attention等更多融合方法
4. 多任务部分，尝试重建，Triplet loss等更多任务
5. GAN部分，不同的辨别模式？cyclegan等更复杂的思路？
6. Domain Adaption？Semi Supervised？辩经（bushi）
7. 和最新的有监督方法对比
8. 和最新的无监督方法对比
9. 完成论文写作，实验数据整理，代码整理，投论文


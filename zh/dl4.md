# 深度学习2：第1部分第4课

### [第4课](http://forums.fast.ai/t/wiki-lesson-4/9402/1)

学生用品：

*   [改善我们学习率的方式](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b)
*   [循环学习率技术](http://teleported.in/posts/cyclic-learning-rate/)
*   [使用重新启动（SGDR）探索随机梯度下降](https://medium.com/38th-street-studios/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e)
*   [使用差异学习率转移学习](https://towardsdatascience.com/transfer-learning-using-differential-learning-rates-638455797f00)
*   [让计算机看得比人类更好](https://medium.com/%40ArjunRajkumar/getting-computers-to-see-better-than-humans-346d96634f73)

![](../img/1_D0WqPCX7RfOL47TOEfkzYg.png)

#### 辍学[04:59]

```
 learn = ConvLearner.pretrained(arch, data, ps=0.5, precompute=True) 
```

*   `precompute=True` ：预先计算从最后一个卷积层出来的激活。 请记住，激活是一个数字，它是根据构成内核/过滤器的一些权重/参数计算出来的，它们会应用于上一层的激活或输入。

```
 learn 
```

```
 _Sequential(_  _(0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)_  _(1): Dropout(p=0.5)_  _(2): Linear(in_features=1024, out_features=512)_  _(3): ReLU()_  _(4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)_  _(5): Dropout(p=0.5)_  _(6): Linear(in_features=512, out_features=120)_  _(7): LogSoftmax()_  _)_ 
```

`learn` - 这将显示我们最后添加的图层。 这些是我们在`precompute=True`时训练的层

（0），（4）： `BatchNorm`将在上一课中介绍

（1），（5）： `Dropout`

（2）： `Linear`层简单地表示矩阵乘法。 这是一个包含1024行和512列的矩阵，因此它将进行1024次激活并吐出512次激活。

（3）： `ReLU` - 只需用零替换负数

（6）： `Linear` - 第二个线性层，从前一个线性层获取512次激活并将它们通过一个新矩阵乘以512乘120并输出120次激活

（7）： `Softmax` - 激活函数，返回最多为1的数字，每个数字在0和1之间：

![](../img/1_PNRoFZeNc0DfGyqsq-S7sA.png)

出于较小的数值精度原因，事实证明最好直接使用softmax的log而不是softmax [ [15:03](https://youtu.be/gbceqO8PpBg%3Ft%3D15m3s) ]。 这就是为什么当我们从模型中得到预测时，我们必须做`np.exp(log_preds)` 。

#### 什么是`Dropout` ，什么是`p` ？ [ [08:17](https://youtu.be/gbceqO8PpBg%3Ft%3D8m17s) ]

```
 _Dropout(p=0.5)_ 
```

![](../img/1_iF4XC8gg608IUouSRI5VrA.png)

如果我们将`p=0.5`压降应用于`Conv2`层，它将如上所示。 我们通过，选择激活，并以50％的几率删除它。 所以`p=0.5`是删除该单元格的概率。 输出实际上并没有太大变化，只是一点点。

随机丢弃一层激活的一半有一个有趣的效果。 需要注意的一件重要事情是，对于每个小批量，我们会丢弃该层中不同的随机半部分激活。 它迫使它不适合。 换句话说，当一个特定的激活只学习那只精确的狗或精确的猫被淘汰时，模型必须尝试找到一个表示，即使随机的一半激活每次被抛弃，它仍然继续工作。

这对于进行现代深度学习工作以及解决泛化问题至关重要。 Geoffrey Hinton和他的同事们提出了这个想法，这个想法受到大脑工作方式的启发。

*   `p=0.01`将丢弃1％的激活。 它根本不会改变任何东西，也不会阻止过度拟合（不是一般化）。
*   `p=0.99`将抛弃99％的激活。 不会过度适应并且非常适合概括，但会破坏你的准确性。
*   默认情况下，第一层为`0.25` ，第二层为`0.5` [17:54]。 如果你发现它过度拟合，就开始碰撞它 - 尝试将全部设置为`0.5` ，仍然过度拟合，尝试`0.7`等。如果你不合适，你可以尝试降低它，但你不太可能需要降低它。
*   ResNet34具有较少的参数，因此它不会过度匹配，但对于像ResNet50这样的更大的架构，您通常需要增加丢失。

你有没有想过为什么验证损失比培训早期的培训损失更好？ [ [12:32](https://youtu.be/gbceqO8PpBg%3Ft%3D12m32s) ]这是因为我们在验证集上运行推理（即进行预测）时关闭了丢失。 我们希望尽可能使用最好的模型。

**问题** ：你是否必须采取任何措施来适应你正在放弃激活的事实？ [ [13:26](https://youtu.be/gbceqO8PpBg%3Ft%3D13m26s) ]我们没有，但是当你说`p=0.5`时，PyTorch会做两件事。 它抛弃了一半的激活，并且它已经存在的所有激活加倍，因此平均激活不会改变。

在Fast.ai中，您可以传入`ps` ，这是所有添加的图层的`p`值。 它不会改变预训练网络中的辍学率，因为它应该已经训练过一些适当的辍学水平：

```
 learn = ConvLearner.pretrained(arch, data, **ps=0.5** , precompute=True) 
```

您可以通过设置`ps=0.`来删除dropout `ps=0.` 但即使在几个时代之后，我们开始大规模过度拟合（训练损失«验证损失）：

```
 [2. **0.3521** **0.55247** 0.84189] 
```

当`ps=0.` ，dropout图层甚至没有添加到模型中：

```
 Sequential(  (0): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)  (1): Linear(in_features=4096, out_features=512)  (2): ReLU()  (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)  (4): Linear(in_features=512, out_features=120)  (5): LogSoftmax()  ) 
```

你可能已经注意到，它已经添加了两个`Linear`层[ [16:19](https://youtu.be/gbceqO8PpBg%3Ft%3D16m19s) ]。 我们不必这样做。 您可以设置`xtra_fc`参数。 注意：您至少需要一个获取卷积层输出（本例中为4096）并将其转换为类数（120个品种）的一个：

```
 learn = ConvLearner.pretrained(arch, data, ps=0., precompute=True,  **xtra_fc=[]** ); learn 
```

```
 _Sequential(_  _(0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)_  _(1): Linear(in_features=1024, out_features=120)_  _(2): LogSoftmax()_  _)_ 
```

```
 learn = ConvLearner.pretrained(arch, data, ps=0., precompute=True,  **xtra_fc=[700, 300]** ); learn 
```

```
 _Sequential(_  _(0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)_  _(1): Linear(in_features=1024, out_features=_ **_700_** _)_  _(2): ReLU()_  _(3): BatchNorm1d(700, eps=1e-05, momentum=0.1, affine=True)_  _(4): Linear(in_features=700, out_features=_ **_300_** _)_  _(5): ReLU()_  _(6): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True)_  _(7): Linear(in_features=300, out_features=120)_  _(8): LogSoftmax()_  _)_ 
```

**问题** ：有没有特定的方法可以确定它是否过度装配？ [ [19:53](https://youtu.be/gbceqO8PpBg%3Ft%3D19m53s) ]。 是的，您可以看到培训损失远低于验证损失。 你无法判断它是否_过度_装修。 零过度拟合通常不是最佳的。 您要做的唯一事情就是降低验证损失，因此您需要尝试一些事情，看看是什么导致验证损失很低。 对于你的特殊问题，你会有一种过度加工的感觉。

**问题** ：为什么平均激活很重要？ [ [21:15](https://youtu.be/gbceqO8PpBg%3Ft%3D21m15s) ]如果我们刚刚删除了一半的激活，那么将它们作为输入的下一次激活也将减半，之后的所有内容。 例如，如果蓬松的耳朵大于0.6，则蓬松的耳朵会蓬松，现在如果它大于0.3则只是蓬松 - 这改变了意义。 这里的目标是删除激活而不改变含义。

**问题** ：我们可以逐层提供不同级别的辍学吗？ [ [22:41](https://youtu.be/gbceqO8PpBg%3Ft%3D22m41s) ]是的，这就是它被称为`ps` ：

```
 learn = ConvLearner.pretrained(arch, data, ps=[0., 0.2],  precompute=True, xtra_fc=[512]); learn 
```

```
 _Sequential(_  _(0): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)_  _(1): Linear(in_features=4096, out_features=512)_  _(2): ReLU()_  _(3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)_  _(4): Dropout(p=0.2)_  _(5): Linear(in_features=512, out_features=120)_  _(6): LogSoftmax()_  _)_ 
```

*   当早期或晚期的图层应该具有不同的辍学量时，没有经验法则。
*   如果有疑问，请为每个完全连接的层使用相同的压差。
*   通常人们只会在最后一个线性层上投入辍学。

**问题** ：为什么要监控损失而不是准确性？ [ [23:53](https://youtu.be/gbceqO8PpBg%3Ft%3D23m53s) ]损失是我们唯一可以看到的验证集和训练集。 正如我们后来所了解的那样，损失是我们实际上正在优化的事情，因此更容易监控和理解这意味着什么。

**问题** ：我们是否需要在添加辍学后调整学习率？[ [24:33](https://youtu.be/gbceqO8PpBg%3Ft%3D24m33s) ]它似乎不足以影响学习率。 理论上，它可能但不足以影响我们。

#### 结构化和时间序列数据[ [25:03](https://youtu.be/gbceqO8PpBg%3Ft%3D25m3s) ]

[笔记本](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) / [Kaggle](https://www.kaggle.com/c/rossmann-store-sales)

![](../img/1_-yc7uZaE44dDVOB850I9-A.png)

列有两种类型：

*   分类 - 它有许多“级别”，例如StoreType，Assortment
*   连续 - 它有一个数字，其中数字的差异或比率具有某种含义，例如竞争距离

```
 cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day',  'StateHoliday', 'CompetitionMonthsOpen', 'Promo2Weeks',  'StoreType', 'Assortment', 'PromoInterval',  'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State',  'Week', 'Events', 'Promo_fw', 'Promo_bw',  'StateHoliday_fw', 'StateHoliday_bw',  'SchoolHoliday_fw', 'SchoolHoliday_bw'] 
```

```
 contin_vars = ['CompetitionDistance', 'Max_TemperatureC',  'Mean_TemperatureC', 'Min_TemperatureC',  'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',  'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',  'CloudCover', 'trend', 'trend_DE',  'AfterStateHoliday', 'BeforeStateHoliday', 'Promo',  'SchoolHoliday'] 
```

```
 n = len(joined); n 
```

*   数字，如`Year` ， `Month` ，虽然我们可以将它们视为连续的，但我们没有必要。 如果我们决定将`Year`作为一个分类变量，我们告诉我们的神经网络，对于`Year` （2000,2001,2002）的每个不同“级别”，你可以完全不同地对待它; 在哪里 - 如果我们说它是连续的，它必须提出某种平滑的功能来适应它们。 通常情况下，实际上是连续的但没有很多不同的级别（例如`Year` ， `DayOfWeek` ），通常将它们视为分类更好。
*   选择分类变量和连续变量是您要做出的建模决策。 总之，如果它在数据中是分类的，则必须是分类的。 如果它在数据中是连续的，您可以选择是在模型中使其连续还是分类。
*   一般来说，浮点数难以分类，因为有很多级别（我们将级别数称为“ **基数** ” - 例如，星期几变量的基数为7）。

**问题** ：你有没有对连续变量进行分类？[ [31:02](https://youtu.be/gbceqO8PpBg%3Ft%3D31m2s) ] Jeremy不会对变量进行分类，但我们可以做的一件事，比如最高温度，分为0-10,10-20,20-30，然后调用分类。 有趣的是，上周刚发表一篇论文，其中一组研究人员发现有时候分组可能会有所帮助。

**问题** ：如果您将年份作为一个类别，当模型遇到一个前所未有的年份时会发生什么？ [ [31:47](https://youtu.be/gbceqO8PpBg%3Ft%3D31m47s) ]我们会到达那里，但简短的回答是，它将被视为一个未知类别。 熊猫有一个特殊的类别叫做未知，如果它看到一个以前没见过的类别，它会被视为未知。

```
 for v in cat_vars:  joined[v] = joined[v].astype('category').cat.as_ordered() 
```

```
 for v in contin_vars:  joined[v] = joined[v].astype('float32') 
```

```
 dep = 'Sales'  joined = joined[cat_vars+contin_vars+[dep, 'Date']].copy() 
```

*   循环遍历`cat_vars`并将适用的数据框列转换为分类列。
*   循环通过`contin_vars`并将它们设置为`float32` （32位浮点），因为这是PyTorch所期望的。

#### 从一个小样本开始[ [34:29](https://youtu.be/gbceqO8PpBg%3Ft%3D34m29s) ]

```
 idxs = get_cv_idxs(n, val_pct=150000/n)  joined_samp = joined.iloc[idxs].set_index("Date")  samp_size = len(joined_samp); samp_size 
```

![](../img/1_dHlXaLjRQSGyrG9pGkWMMQ.png)

这是我们的数据。 即使我们将一些列设置为“类别”（例如'StoreType'，'Year'），Pandas仍然在笔记本中显示为字符串。

```
 df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)  yl = np.log(y) 
```

`proc_df` （进程数据框） - Fast.ai中的一个函数，它执行以下操作：

1.  拉出因变量，将其放入单独的变量中，并从原始数据框中删除它。 换句话说， `df`没有`Sales`列， `y`只包含`Sales`列。
2.  `do_scale` ：神经网络真的希望所有输入数据都在零左右，标准偏差大约为1.因此，我们取数据，减去均值，然后除以标准偏差即可。 它返回一个特殊对象，用于跟踪它用于该规范化的均值和标准偏差，因此您可以稍后对测试集执行相同操作（ `mapper` ）。
3.  它还处理缺失值 - 对于分类变量，它变为ID：0，其他类别变为1,2,3等。 对于连续变量，它用中位数替换缺失值，并创建一个新的布尔列，说明它是否丢失。

![](../img/1_Zs6ASJF8iaAe3cduCmLYKw.png)

在处理之后，2014年例如变为2，因为分类变量已经被从零开始的连续整数替换。 原因是，我们将在稍后将它们放入矩阵中，并且当它可能只是两行时，我们不希望矩阵长度为2014行。

现在我们有一个数据框，它不包含因变量，一切都是数字。 这就是我们需要深入学习的地方。 查看机器学习课程了解更多详情。 机器学习课程中涉及的另一件事是验证集。 在这种情况下，我们需要预测未来两周的销售情况，因此我们应该创建一个验证集，这是我们培训集的最后两周：

```
 val_idx = np.flatnonzero((df.index<=datetime.datetime(2014,9,17)) &  (df.index>=datetime.datetime(2014,8,1))) 
```

*   [如何（以及为什么）创建一个好的验证集](http://www.fast.ai/2017/11/13/validation-sets/)

#### 让我们直接进入深度学习行动[ [39:48](https://youtu.be/gbceqO8PpBg%3Ft%3D39m48s) ]

对于任何Kaggle比赛，重要的是您要充分了解您的指标 - 您将如何评判。 在[本次比赛中](https://www.kaggle.com/c/rossmann-store-sales) ，我们将根据均方根百分比误差（RMSPE）进行判断。

![](../img/1_a7mJ5VCeuAxagGrHOq6ekQ.png)

```
 def inv_y(a): return np.exp(a) 
```

```
 def exp_rmspe(y_pred, targ):  targ = inv_y(targ)  pct_var = (targ - inv_y(y_pred))/targ  return math.sqrt((pct_var**2).mean()) 
```

```
 max_log_y = np.max(yl)  y_range = (0, max_log_y*1.2) 
```

*   当您获取数据的日志时，获得均方根误差实际上会得到均方根百分比误差。

```
 md = **ColumnarModelData.from_data_frame** (PATH, val_idx, df,  yl.astype(np.float32), cat_flds=cat_vars, bs=128,  test_df=df_test) 
```

*   按照惯例，我们将从创建模型数据对象开始，该对象具有内置于其中的验证集，训练集和可选测试集。 从那以后，我们将获得一个学习者，然后我们可以选择调用`lr_find` ，然后调用`learn.fit`等等。
*   这里的区别是我们没有使用`ImageClassifierData.from_csv`或`.from_paths` ，我们需要一种名为`.from_paths`的不同类型的模型数据，我们调用`from_data_frame` 。
*   `PATH` ：指定存储模型文件的位置等
*   `val_idx` ：我们要放入验证集的行的索引列表
*   `df` ：包含自变量的数据框
*   `yl` ：我们取了`proc_df`返回的因变量`y`并记录了它的日志（即`np.log(y)` ）
*   `cat_flds` ： `cat_flds`哪些列视为分类。 请记住，到目前为止，一切都是一个数字，所以除非我们指定，否则它们将全部视为连续的。

现在我们有一个熟悉的标准模型数据对象，包含`train_dl` ， `val_dl` ， `train_ds` ， `val_ds`等。

```
 m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),  0.04, 1, [1000,500], [0.001,0.01],  y_range=y_range) 
```

*   在这里，我们要求它创建一个适合我们的模型数据的学习者。
*   `0.04` ：使用多少辍学
*   `[1000,500]` ： `[1000,500]`激活多少次
*   `[0.001,0.01]` ：在后续层使用多少辍学者

#### 关键新概念：嵌入[ [45:39](https://youtu.be/gbceqO8PpBg%3Ft%3D45m39s) ]

我们暂时忘记分类变量：

![](../img/1_T604NRtHHBkBWFvWoovlUw.png)

请记住，您永远不想将ReLU放在最后一层，因为softmax需要负数来创建低概率。

#### **完全连接神经网络的简单视图[** [**49:13**](https://youtu.be/gbceqO8PpBg%3Ft%3D49m13s) **]：**

![](../img/1_5D0_nDy0K0QLKFHTD07gcQ.png)

对于回归问题（不是分类），您甚至可以跳过softmax图层。

#### 分类变量[ [50:49](https://youtu.be/gbceqO8PpBg%3Ft%3D50m49s) ]

我们创建一个7行的新矩阵和我们选择的列数（例如4）并用浮点数填充它。 要使用连续变量将“星期日”添加到我们的等级1张量中，我们会查看此矩阵，它将返回4个浮点数，并将它们用作“星期日”。

![](../img/1_cAgCy5HfD0rvPDg2dQITeg.png)

最初，这些数字是随机的。 但我们可以通过神经网络将它们更新，并以减少损失的方式更新它们。 换句话说，这个矩阵只是我们神经网络中的另一组权重。 这种类型的**矩阵**称为“ **嵌入矩阵** ”。 嵌入矩阵是我们从该类别的零和最大级别之间的整数开始的。 我们索引矩阵以找到一个特定的行，然后将它追加到我们所有的连续变量中，之后的所有内容与之前的相同（线性→ReLU→等）。

**问题** ：这4个数字代表什么？[ [55:12](https://youtu.be/gbceqO8PpBg%3Ft%3D55m12s) ]当我们看协同过滤时，我们会更多地了解这一点，但就目前而言，它们只是我们正在学习的参数，最终会给我们带来很大的损失。 我们稍后会发现这些特定的参数通常是人类可解释的并且非常有趣，但这是副作用。

**问题** ：您对嵌入矩阵的维数有很好的启发式吗？ [ [55:57](https://youtu.be/gbceqO8PpBg%3Ft%3D55m57s) ]我确实做到了！ 让我们来看看。

```
 cat_sz = [(c, len(joined_samp[c].cat.categories)+1)  **for** c **in** cat_vars]  cat_sz 
```

```
 _[('Store', 1116),_  _('DayOfWeek', 8),_  _('Year', 4),_  _('Month', 13),_  _('Day', 32),_  _('StateHoliday', 3),_  _('CompetitionMonthsOpen', 26),_  _('Promo2Weeks', 27),_  _('StoreType', 5),_  _('Assortment', 4),_  _('PromoInterval', 4),_  _('CompetitionOpenSinceYear', 24),_  _('Promo2SinceYear', 9),_  _('State', 13),_  _('Week', 53),_  _('Events', 22),_  _('Promo_fw', 7),_  _('Promo_bw', 7),_  _('StateHoliday_fw', 4),_  _('StateHoliday_bw', 4),_  _('SchoolHoliday_fw', 9),_  _('SchoolHoliday_bw', 9)]_ 
```

*   以下是每个分类变量及其基数的列表。
*   即使原始数据中没有缺失值，您仍然应该留出一个未知的，以防万一。
*   确定嵌入大小的经验法则是基数大小除以2，但不大于50。

```
 emb_szs = [(c, min(50, (c+1)//2)) **for** _,c **in** cat_sz]  emb_szs 
```

```
 _[(1116, 50),_  _(8, 4),_  _(4, 2),_  _(13, 7),_  _(32, 16),_  _(3, 2),_  _(26, 13),_  _(27, 14),_  _(5, 3),_  _(4, 2),_  _(4, 2),_  _(24, 12),_  _(9, 5),_  _(13, 7),_  _(53, 27),_  _(22, 11),_  _(7, 4),_  _(7, 4),_  _(4, 2),_  _(4, 2),_  _(9, 5),_  _(9, 5)]_ 
```

然后将嵌入大小传递给学习者：

```
 m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1,  [1000,500], [0.001,0.01], y_range=y_range) 
```

**问题** ：有没有办法初始化嵌入矩阵除了随机？ [ [58:14](https://youtu.be/gbceqO8PpBg%3Ft%3D58m14s) ]我们可能会在课程的后期讨论预训练，但基本的想法是，如果罗斯曼的其他人已经训练了一个神经网络来预测奶酪销售，你也可以从他们的嵌入矩阵开始商店预测酒类销售。 例如，在Pinterest和Instacart就会发生这种情况。 Instacart使用这种技术来路由他们的购物者，Pinterest使用它来决定在网页上显示什么。 他们嵌入了在组织中共享的产品/商店矩阵，因此人们无需培训新的产品/商店。

**问题** ：使用嵌入矩阵优于单热编码有什么好处？ [ [59:23](https://youtu.be/gbceqO8PpBg%3Ft%3D59m23s) ]对于上面一周的例子，我们可以很容易地传递7个数字（例如星期日的[ [0,1,0,0,0,0,0](https://youtu.be/gbceqO8PpBg%3Ft%3D59m23s) ]），而不是4个数字。 这也是一个浮动列表，这将完全起作用 - 这就是一般来说，分类变量多年来一直用于统计（称为“虚拟变量”）。 问题是，星期日的概念只能与单个浮点数相关联。 所以它得到了这种线性行为 - 它说周日或多或少只是一件事。 通过嵌入，星期日是四维空间的概念。 我们倾向于发现的是这些嵌入向量倾向于获得这些丰富的语义概念。 例如，如果事实证明周末有不同的行为，您往往会看到周六和周日会有更高的特定数字。

> 通过具有更高的维度向量而不仅仅是单个数字，它为深度学习网络提供了学习这些丰富表示的机会。

嵌入的想法是所谓的“分布式表示” - 神经网络的最基本概念。 这就是神经网络中的概念具有很难解释的高维表示的想法。 这个向量中的这些数字甚至不必只有一个含义。 它可能意味着一件事，如果这个是低的，一个是高的，如果那个是高的那个，而另一个是低的，因为它正在经历这个丰富的非线性函数。 正是这种丰富的表现形式使它能够学习这种有趣的关系。

**问题** ：嵌入是否适合某些类型的变量？ [ [01:02:45](https://youtu.be/gbceqO8PpBg%3Ft%3D1h2m45s) ]嵌入适用于任何分类变量。 它唯一不能很好地工作的是基数太高的东西。 如果您有600,000行且变量有600,000个级别，那么这不是一个有用的分类变量。 但总的来说，本次比赛的第三名获胜者确实认为所有基因都不是太高，他们都把它们都视为绝对的。 好的经验法则是，如果你可以创建一个分类变量，你也可以这样，因为它可以学习这种丰富的分布式表示; 如果你把它留在连续的地方，它最能做的就是试着找到一个适合它的单一功能形式。

#### 场景背后的矩阵代数[ [01:04:47](https://youtu.be/gbceqO8PpBg%3Ft%3D1h4m47s) ]

查找具有索引的嵌入与在单热编码向量和嵌入矩阵之间进行矩阵乘积相同。 但这样做非常低效，因此现代库实现这一点，即采用整数并查看数组。

![](../img/1_psxpwtr5bw55lKxVV_y81w.png)

**问题** ：您能否触及使用日期和时间作为分类以及它如何影响季节性？ [ [01:06:59](https://youtu.be/gbceqO8PpBg%3Ft%3D1h6m59s) ]有一个名为`add_datepart`的Fast.ai函数，它接受数据框和列名。 它可以选择从数据框中删除该列，并将其替换为代表该日期的所有有用信息的大量列，例如星期几，日期，月份等等（基本上是Pandas给我们的所有内容）。

```
 add_datepart(weather, "Date", drop=False)  add_datepart(googletrend, "Date", drop=False)  add_datepart(train, "Date", drop=False)  add_datepart(test, "Date", drop=False) 
```

![](../img/1_OJQ53sO6WXh0C-rzw1QyJg.png)

因此，例如，星期几现在变为八行四列嵌入矩阵。 从概念上讲，这允许我们的模型创建一些有趣的时间序列模型。 如果有一个七天周期的周期在周一和周三上升，但仅限于每天和仅在柏林，它可以完全这样做 - 它拥有它需要的所有信息。 这是处理时间序列的绝佳方式。 您只需确保时间序列中的循环指示符作为列存在。 如果你没有一个名为day of week的列，那么神经网络很难学会做mod 7并在嵌入矩阵中查找。 这不是不可能，但真的很难。 如果你预测旧金山的饮料销售，你可能想要一个AT＆T公园球赛开始时的清单，因为这会影响到SoMa有多少人在喝啤酒。 因此，您需要确保基本指标或周期性在您的数据中，并且只要它们在那里，神经网络将学会使用它们。

#### 学习者[ [01:10:13](https://youtu.be/gbceqO8PpBg%3Ft%3D1h10m13s) ]

```
 m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1,  [1000,500], [0.001,0.01], y_range=y_range)  lr = 1e-3 
```

*   `emb_szs` ：嵌入大小
*   `len(df.columns)-len(cat_vars)` ：数据框中连续变量的数量
*   `0.04` ：嵌入矩阵有自己的丢失，这是辍学率
*   `1` ：我们想要创建多少输出（最后一个线性层的输出）
*   `[1000, 500]` ：第一线性层和第二线性层中的激活次数
*   `[0.001, 0.01]` ：第一线性层和第二线性层中的脱落
*   `y_range` ：我们暂时不担心

```
 m.fit(lr, 3, metrics=[exp_rmspe]) 
```

```
 _A Jupyter Widget_ 
```

```
 _[ 0\. 0.02479 0.02205_ **_0.19309_** _]_  _[ 1\. 0.02044 0.01751_ **_0.18301_** _]_  _[ 2\. 0.01598 0.01571_ **_0.17248_** _]_ 
```

*   `metrics` ：这是一个自定义指标，它指定在每个纪元结束时调用的函数并打印出结果

```
 m.fit(lr, 1, metrics=[exp_rmspe], cycle_len=1) 
```

```
 _[ 0\. 0.00676 0.01041 0.09711]_ 
```

通过使用所有训练数据，我们实现了大约0.09711的RMSPE。 公共领导委员会和私人领导委员会之间存在很大差异，但我们肯定是本次竞赛的最高端。

所以这是一种处理时间序列和结构化数据的技术。 有趣的是，与使用这种技术的组（ [分类变量的实体嵌入](https://arxiv.org/abs/1604.06737) ）相比，第二名获胜者做了更多的特征工程。 本次比赛的获胜者实际上是物流销售预测的主题专家，因此他们有自己的代码来创建大量的功能。 Pinterest的人们为建议建立了一个非常相似的模型也表示，当他们从梯度增强机器转向深度学习时，他们的功能工程设计更少，而且模型更简单，需要的维护更少。 因此，这是使用这种深度学习方法的一大好处 - 您可以获得最先进的结果，但工作量却少得多。

**问题** ：我们是否正在使用任何时间序列？ [ [01:15:01](https://youtu.be/gbceqO8PpBg%3Ft%3D1h15m1s) ]间接地，是的。 正如我们刚刚看到的那样，我们的列中有一周中的一周，一年中的一些等，其中大多数都被视为类别，因此我们正在构建一月，周日等的分布式表示。 我们没有使用任何经典的时间序列技术，我们所做的只是在神经网络中真正完全连接的层。 嵌入矩阵能够以比任何标准时间序列技术更丰富的方式处理诸如星期几周期性之类的事情。

关于图像模型和这个模型之间差异的**问题** [ [01:15:59](https://youtu.be/gbceqO8PpBg%3Ft%3D1h15m59s) ]：我们调用`get_learner`的方式有所不同。 在成像中我们只是做了`Learner.trained`并传递数据：

```
 learn = ConvLearner.pretrained(arch, data, ps=0., precompute=True) 
```

对于这些类型的模型，事实上对于许多模型，我们构建的模型取决于数据。 在这种情况下，我们需要知道我们有什么嵌入矩阵。 所以在这种情况下，数据对象创建了学习者（颠倒到我们之前看到的）：

```
 m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1,  [1000,500], [0.001,0.01], y_range=y_range) 
```

**步骤摘要** （如果你想将它用于你自己的数据集）[ [01:17:56](https://youtu.be/gbceqO8PpBg%3Ft%3D1h17m56s) ]：

**第1步** 。 列出分类变量名称，并列出连续变量名称，并将它们放在Pandas数据框中

**第2步** 。 在验证集中创建所需的行索引列表

**第3步** 。 调用这段确切的代码：

```
 md = ColumnarModelData.from_data_frame(PATH, val_idx, df,  yl.astype(np.float32), cat_flds=cat_vars, bs=128,  test_df=df_test) 
```

**第4步** 。 创建一个列表，列出每个嵌入矩阵的大小

**第5步** 。 调用`get_learner` - 您可以使用这些确切的参数开头：

```
 m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1,  [1000,500], [0.001,0.01], y_range=y_range) 
```

**第6步** 。 打电话给`m.fit`

**问题** ：如何对此类数据使用数据扩充，以及丢失如何工作？ [ [01:18:59](https://youtu.be/gbceqO8PpBg%3Ft%3D1h18m59s) ]不知道。 Jeremy认为它必须是针对特定领域的，但他从未见过任何论文或业内任何人使用结构化数据和深度学习进行数据增强。 他认为可以做到但没有看到它完成。 辍学者正在做什么与以前完全一样。

**问题** ：缺点是什么？ 几乎没有人使用这个。 为什么不？ [ [01:20:41](https://youtu.be/gbceqO8PpBg%3Ft%3D1h20m41s) ]基本上答案就像我们之前讨论过的那样，学术界没有人差不多正在研究这个问题，因为这不是人们发表的内容。 结果，人们可以看到的并没有很好的例子，并且说“哦，这是一种运作良好的技术，让我们的公司实施它”。 但也许同样重要的是，到目前为止，使用这个Fast.ai库，还没有任何方法可以方便地进行。 如果您想要实现其中一个模型，则必须自己编写所有自定义代码。 有很多大的商业和科学机会来使用它并解决以前未能很好解决的问题。

### 自然语言处理[ [01:23:37](https://youtu.be/gbceqO8PpBg%3Ft%3D1h23m37s) ]

最具前瞻性的深度学习领域，它落后于计算机视觉两三年。 软件的状态和一些概念远没有计算机视觉那么成熟。 您在NLP中找到的一件事是您可以解决的特殊问题，并且它们具有特定的名称。 NLP中存在一种称为“语言建模”的特殊问题，它有一个非常具体的定义 - 它意味着建立一个模型，只要给出一个句子的几个单词，你能预测下一个单词将会是什么。

#### 语言建模[ [01:25:48](https://youtu.be/gbceqO8PpBg%3Ft%3D1h25m48s) ]

[笔记本](https://github.com/fastai/fastai/blob/master/courses/dl1/lang_model-arxiv.ipynb)

这里我们有来自arXiv（arXiv.org）的18个月的论文，这是一个例子：

```
 ' '.join(md.trn_ds[0].text[:150]) 
```

```
 _'<cat> csni <summ> the exploitation of mm - wave bands is one of the key - enabler for 5 g mobile \n radio networks ._ _however , the introduction of mm - wave technologies in cellular \n networks is not straightforward due to harsh propagation conditions that limit \n the mm - wave access availability ._ _mm - wave technologies require high - gain antenna \n systems to compensate for high path loss and limited power ._ _as a consequence , \n directional transmissions must be used for cell discovery and synchronization \n processes : this can lead to a non - negligible access delay caused by the \n exploration of the cell area with multiple transmissions along different \n directions ._ _\n the integration of mm - wave technologies and conventional wireless access \n networks with the objective of speeding up the cell search process requires new \n'_ 
```

*   `&lt;cat&gt;` - 论文的类别。 CSNI是计算机科学和网络
*   `&lt;summ&gt;` - 论文摘要

以下是训练有素的语言模型的输出结果。 我们做了简单的小测试，在这些测试中你传递了一些启动文本，看看模型认为下一步应该是什么：

```
 sample_model(m, "<CAT> csni <SUMM> algorithms that") 
```

```
 _...use the same network as a single node are not able to achieve the same performance as the traditional network - based routing algorithms ._ _in this paper , we propose a novel routing scheme for routing protocols in wireless networks ._ _the proposed scheme is based ..._ 
```

它通过阅读arXiv论文得知，正在写关于计算机网络的人会这样说。 记住，它开始根本不懂英语。 它开始时是一个嵌入矩阵，用于英语中每个随机的单词。 通过阅读大量的arXiv论文，它学到了什么样的单词跟随他人。

在这里，我们尝试将类别指定为计算机视觉：

```
 sample_model(m, "<CAT> cscv <SUMM> algorithms that") 
```

```
 _...use the same data to perform image classification are increasingly being used to improve the performance of image classification algorithms ._ _in this paper , we propose a novel method for image classification using a deep convolutional neural network ( cnn ) ._ _the proposed method is ..._ 
```

它不仅学会了如何写好英语，而且在你说出“卷积神经网络”之后，你应该使用括号来指定首字母缩略词“（CNN）”。

```
 sample_model(m,"<CAT> cscv <SUMM> algorithms. <TITLE> on ") 
```

```
 ...the performance of deep learning for image classification <eos> 
```

```
 sample_model(m,"<CAT> csni <SUMM> algorithms. <TITLE> on ") 
```

```
 ...the performance of wireless networks <eos> 
```

```
 sample_model(m,"<CAT> cscv <SUMM> algorithms. <TITLE> towards ") 
```

```
 ...a new approach to image classification <eos> 
```

```
 sample_model(m,"<CAT> csni <SUMM> algorithms. <TITLE> towards ") 
```

```
 ...a new approach to the analysis of wireless networks <eos> 
```

A language model can be incredibly deep and subtle, so we are going to try and build that — not because we care about this at all, but because we are trying to create a pre-trained model which is used to do some other tasks. For example, given an IMDB movie review, we will figure out whether they are positive or negative. It is a lot like cats vs. dogs — a classification problem. So we would really like to use a pre-trained network which at least knows how to read English. So we will train a model that predicts a next word of a sentence (ie language model), and just like in computer vision, stick some new layers on the end and ask it to predict whether something is positive or negative.

#### IMDB [ [1:31:11](https://youtu.be/gbceqO8PpBg%3Ft%3D1h31m11s) ]

[笔记本](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb)

What we are going to do is to train a language model, making that the pre-trained model for a classification model. In other words, we are trying to leverage exactly what we learned in our computer vision which is how to do fine-tuning to create powerful classification models.

**Question** : why would doing directly what you want to do not work? [ [01:31:34](https://youtu.be/gbceqO8PpBg%3Ft%3D1h31m34s) ] It just turns out it doesn't empirically. There are several reasons. First of all, we know fine-tuning a pre-trained network is really powerful. So if we can get it to learn some related tasks first, then we can use all that information to try and help it on the second task. The other is IMDB movie reviews are up to a thousands words long. So after reading a thousands words knowing nothing about how English is structured or concept of a word or punctuation, all you get is a 1 or a 0 (positive or negative). Trying to learn the entire structure of English and then how it expresses positive and negative sentiments from a single number is just too much to expect.

**Question** : Is this similar to Char-RNN by Karpathy? [ [01:33:09](https://youtu.be/gbceqO8PpBg%3Ft%3D1h33m9s) ] This is somewhat similar to Char-RNN which predicts the next letter given a number of previous letters. Language model generally work at a word level (but they do not have to), and we will focus on word level modeling in this course.

**Question** : To what extent are these generated words/sentences actual copies of what it found in the training set? [ [01:33:44](https://youtu.be/gbceqO8PpBg%3Ft%3D1h33m44s) ] Words are definitely words it has seen before because it is not a character level so it can only give us the word it has seen before. Sentences, there are rigorous ways of doing it but the easiest would be by looking at examples like above, you get a sense of it. Most importantly, when we train the language model, we will have a validation set so that we are trying to predict the next word of something that has never seen before. There are tricks to using language models to generate text like [beam search](http://forums.fast.ai/t/tricks-for-using-language-models-to-generate-text/8127/2) .

Use cases of text classification:

*   For hedge fund, identify things in articles or Twitter that caused massive market drops in the past.
*   Identify customer service queries which tend to be associated with people who cancel their contracts in the next month
*   Organize documents into whether they are part of legal discovery or not.

```
 from fastai.learner import * 
```

```
 import torchtext  from torchtext import vocab, data  from torchtext.datasets import language_modeling 
```

```
 from fastai.rnn_reg import *  from fastai.rnn_train import *  from fastai.nlp import *  from fastai.lm_rnn import * 
```

```
 import dill as pickle 
```

*   `torchtext` — PyTorch's NLP library

#### Data [ [01:37:05](https://youtu.be/gbceqO8PpBg%3Ft%3D1h37m5s) ]

IMDB [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

```
 PATH = 'data/aclImdb/' 
```

```
 TRN_PATH = 'train/all/'  VAL_PATH = 'test/all/'  TRN = f'{PATH}{TRN_PATH}'  VAL = f'{PATH}{VAL_PATH}' 
```

```
 %ls {PATH} 
```

```
 imdbEr.txt imdb.vocab models/ README test/ tmp/ train/ 
```

We do not have separate test and validation in this case. Just like in vision, the training directory has bunch of files in it:

```
 trn_files = !ls {TRN}  trn_files[:10]  ['0_0.txt', 
 '0_3.txt', 
 '0_9.txt', 
 '10000_0.txt', 
 '10000_4.txt', 
 '10000_8.txt', 
 '1000_0.txt', 
 '10001_0.txt', 
 '10001_10.txt', 
 '10001_4.txt'] 
```

```
 review = !cat {TRN}{trn_files[6]}  review[0] 
```

```
 "I have to say when a name like Zombiegeddon and an atom bomb on the front cover I was expecting a flat out chop-socky fung-ku, but what I got instead was a comedy. So, it wasn't quite was I was expecting, but I really liked it anyway! The best scene ever was the main cop dude pulling those kids over and pulling a Bad Lieutenant on them!! I was laughing my ass off. I mean, the cops were just so bad! And when I say bad, I mean The Shield Vic Macky bad. But unlike that show I was laughing when they shot people and smoked dope.<br /><br />Felissa Rose...man, oh man. What can you say about that hottie. She was great and put those other actresses to shame. She should work more often!!!!! I also really liked the fight scene outside of the building. That was done really well. Lots of fighting and people getting their heads banged up. FUN! Last, but not least Joe Estevez and William Smith were great as the...well, I wasn't sure what they were, but they seemed to be having fun and throwing out lines. I mean, some of it didn't make sense with the rest of the flick, but who cares when you're laughing so hard! All in all the film wasn't the greatest thing since sliced bread, but I wasn't expecting that. It was a Troma flick so I figured it would totally suck. It's nice when something surprises you but not totally sucking.<br /><br />Rent it if you want to get stoned on a Friday night and laugh with your buddies. Don't rent it if you are an uptight weenie or want a zombie movie with lots of flesh eating.<br /><br />PS Uwe Boil was a nice touch." 
```

Now we will check how many words are in the dataset:

```
 !find {TRN} -name '*.txt' | xargs cat | wc -w 
```

```
 17486581 
```

```
 !find {VAL} -name '*.txt' | xargs cat | wc -w 
```

```
 5686719 
```

Before we can do anything with text, we have to turn it into a list of tokens. Token is basically like a word. Eventually we will turn them into a list of numbers, but the first step is to turn it into a list of words — this is called “tokenization” in NLP. A good tokenizer will do a good job of recognizing pieces in your sentence. Each separated piece of punctuation will be separated, and each part of multi-part word will be separated as appropriate. Spacy does a lot of NLP stuff, and it has the best tokenizer Jeremy knows. So Fast.ai library is designed to work well with the Spacey tokenizer as with torchtext.

#### Creating a field [ [01:41:01](https://youtu.be/gbceqO8PpBg%3Ft%3D1h41m1s) ]

A field is a definition of how to pre-process some text.

```
 TEXT = data.Field(lower= True , tokenize=spacy_tok) 
```

*   `lower=True` — lowercase the text
*   `tokenize=spacy_tok` — tokenize with `spacy_tok`

Now we create the usual Fast.ai model data object:

```
 bs=64; bptt=70 
```

```
 FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)  md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs,  bptt=bptt, min_freq=10) 
```

*   `PATH` : as per usual where the data is, where to save models, etc
*   `TEXT` : torchtext's Field definition
*   `**FILES` : list of all of the files we have: training, validation, and test (to keep things simple, we do not have a separate validation and test set, so both points to validation folder)
*   `bs` : batch size
*   `bptt` : Back Prop Through Time. It means how long a sentence we will stick on the GPU at once
*   `min_freq=10` : In a moment, we are going to be replacing words with integers (a unique index for every word). If there are any words that occur less than 10 times, just call it unknown.

After building our `ModelData` object, it automatically fills the `TEXT` object with a very important attribute: `TEXT.vocab` . This is a _vocabulary_ , which stores which unique words (or _tokens_ ) have been seen in the text, and how each word will be mapped to a unique integer id.

```
 # 'itos': 'int-to-string'  TEXT.vocab.itos[:12] 
```

```
 ['<unk>', '<pad>', 'the', ',', '.', 'and', 'a', 'of', 'to', 'is', 'it', 'in'] 
```

```
 # 'stoi': 'string to int'  TEXT.vocab.stoi['the'] 
```

```
 _2_ 
```

`itos` is sorted by frequency except for the first two special ones. Using `vocab` , torchtext will turn words into integer IDs for us :

```
 md.trn_ds[0].text[:12] 
```

```
 ['i', 
 'have', 
 'always', 
 'loved', 
 'this', 
 'story', 
 '-', 
 'the', 
 'hopeful', 
 'theme', 
 ',', 
 'the'] 
```

```
 TEXT.numericalize([md.trn_ds[0].text[:12]]) 
```

```
 Variable containing: 
 _12_  _35_  _227_  _480_  _13_  _76_  _17_  _2_ 
 7319 
 _769_  _3_  _2_ 
 [torch.cuda.LongTensor of size 12x1 (GPU 0)] 
```

**Question** : Is it common to do any stemming or lemma-tizing? [ [01:45:47](https://youtu.be/gbceqO8PpBg%3Ft%3D1h45m47s) ] Not really, no. Generally tokenization is what we want. To keep it as general as possible, we want to know what is coming next so whether it is future tense or past tense or plural or singular, we don't really know which things are going to be interesting and which are not, so it seems that it is generally best to leave it alone as much as possible.

**Question** : When dealing with natural language, isn't context important? Why are we tokenizing and looking at individual word? [ [01:46:38](https://youtu.be/gbceqO8PpBg%3Ft%3D1h46m38s) ] No, we are not looking at individual word — they are still in order. Just because we replaced I with a number 12, they are still in that order. There is a different way of dealing with natural language called “bag of words” and they do throw away the order and context. In the Machine Learning course, we will be learning about working with bag of words representations but my belief is that they are no longer useful or in the verge of becoming no longer useful. We are starting to learn how to use deep learning to use context properly.

#### Batch size and BPTT [ [01:47:40](https://youtu.be/gbceqO8PpBg%3Ft%3D1h47m40s) ]

What happens in a language model is even though we have lots of movie reviews, they all get concatenated together into one big block of text. So we predict the next word in this huge long thing which is all of the IMDB movie reviews concatenated together.

![](../img/1_O-Kq1qtgZmrShbKhaN3fTg.png)

*   We split up the concatenated reviews into batches. In this case, we will split it to 64 sections
*   We then move each section underneath the previous one, and transpose it.
*   We end up with a matrix which is 1 million by 64\.
*   We then grab a little chunk at time and those chunk lengths are **approximately** equal to BPTT. Here, we grab a little 70 long section and that is the first thing we chuck into our GPU (ie the batch).

```
 next(iter(md.trn_dl)) 
```

```
 (Variable containing: 
 12 567 3 ... 2118 4 2399 
  _35 7 33_  ... 6 148 55 
 227 103 533 ... 4892 31 10 
 ... ⋱ ... 
 19 8879 33 ... 41 24 733 
 552 8250 57 ... 219 57 1777 
 5 19 2 ... 3099 8 48 
 [torch.cuda.LongTensor of size 75x64 (GPU 0)], Variable containing: 
 **_35_**  **_7_**  **_33_** 
 ⋮ 
 _22_ 
 3885 
 21587 
 [torch.cuda.LongTensor of size 4800 (GPU 0)]) 
```

*   We grab our first training batch by wrapping data loader with `iter` then calling `next` .
*   We got back a 75 by 64 tensor (approximately 70 rows but not exactly)
*   A neat trick torchtext does is to randomly change the `bptt` number every time so each epoch it is getting slightly different bits of text — similar to shuffling images in computer vision. We cannot randomly shuffle the words because they need to be in the right order, so instead, we randomly move their breakpoints a little bit.
*   The target value is also 75 by 64 but for minor technical reasons it is flattened out into a single vector.

**Question** : Why not split by a sentence? [ [01:53:40](https://youtu.be/gbceqO8PpBg%3Ft%3D1h53m40s) ] Not really. Remember, we are using columns. So each of our column is of length about 1 million, so although it is true that those columns are not always exactly finishing on a full stop, they are so darn long we do not care. Each column contains multiple sentences.

Pertaining to this question, Jeremy found what is in this language model matrix a little mind-bending for quite a while, so do not worry if it takes a while and you have to ask a thousands questions.

#### Create a model [ [01:55:46](https://youtu.be/gbceqO8PpBg%3Ft%3D1h55m46s) ]

Now that we have a model data object that can fee d us batches, we can create a model. First, we are going to create an embedding matrix.

Here are the: # batches; # unique tokens in the vocab; length of the dataset; # of words

```
 len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text) 
```

```
 (4602, 34945, 1, 20621966) 
```

This is our embedding matrix looks like:

![](../img/1_6EHxqeSYMioiLEQ5ufrf_g.png)

*   It is a high cardinality categorical variable and furthermore, it is the only variable — this is typical in NLP
*   The embedding size is 200 which is much bigger than our previous embedding vectors. Not surprising because a word has a lot more nuance to it than the concept of Sunday. **Generally, an embedding size for a word will be somewhere between 50 and 600.**

```
 em_sz = 200 # size of each embedding vector  nh = 500 # number of hidden activations per layer  nl = 3 # number of layers 
```

Researchers have found that large amounts of _momentum_ (which we'll learn about later) don't work well with these kinds of _RNN_ models, so we create a version of the _Adam_ optimizer with less momentum than its default of `0.9` . Any time you are doing NLP, you should probably include this line:

```
 opt_fn = partial(optim.Adam, betas=(0.7, 0.99)) 
```

Fast.ai uses a variant of the state of the art [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182) developed by Stephen Merity. A key feature of this model is that it provides excellent regularization through [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network) . There is no simple way known (yet!) to find the best values of the dropout parameters below — you just have to experiment…

However, the other parameters ( `alpha` , `beta` , and `clip` ) shouldn't generally need tuning.

```
 learner = md.get_model(opt_fn, em_sz, nh, nl, dropouti=0.05,  dropout=0.05, wdrop=0.1, dropoute=0.02,  dropouth=0.05)  learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)  learner.clip=0.3 
```

*   In the last lecture, we will learn what the architecture is and what all these dropouts are. For now, just know it is the same as per usual, if you try to build an NLP model and you are under-fitting, then decrease all these dropouts, if overfitting, then increase all these dropouts in roughly this ratio. Since this is such a recent paper so there is not a lot of guidance but these ratios worked well — it is what Stephen has been using as well.
*   There is another kind of way we can avoid overfitting that we will talk about in the last class. For now, `learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)` works reliably so all of your NLP models probably want this particular line.
*   `learner.clip=0.3` : when you look at your gradients and you multiply them by the learning rate to decide how much to update your weights by, this will not allow them be more than 0.3\. This is a cool little trick to prevent us from taking too big of a step.
*   Details do not matter too much right now, so you can use them as they are.

**Question** : There are word embedding out there such as Word2vec or GloVe. How are they different from this? And why not initialize the weights with those initially? [ [02:02:29](https://youtu.be/gbceqO8PpBg%3Ft%3D2h2m29s) ] People have pre-trained these embedding matrices before to do various other tasks. They are not called pre-trained models; they are just a pre-trained embedding matrix and you can download them. There is no reason we could not download them. I found that building a whole pre-trained model in this way did not seem to benefit much if at all from using pre-trained word vectors; where else using a whole pre-trained language model made a much bigger difference. Maybe we can combine both to make them a little better still.

**Question:** What is the architecture of the model? [ [02:03:55](https://youtu.be/gbceqO8PpBg%3Ft%3D2h3m55s) ] We will be learning about the model architecture in the last lesson but for now, it is a recurrent neural network using something called LSTM (Long Short Term Memory).

#### Fitting [ [02:04:24](https://youtu.be/gbceqO8PpBg%3Ft%3D2h4m24s) ]

```
 learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2) 
```

```
 learner.save_encoder('adam1_enc') 
```

```
 learner.fit(3e-3, 4, wds=1e-6, cycle_len=10,  cycle_save_name='adam3_10') 
```

```
 learner.save_encoder('adam3_10_enc') 
```

```
 learner.fit(3e-3, 1, wds=1e-6, cycle_len=20,  cycle_save_name='adam3_20') 
```

```
 learner.load_cycle('adam3_20',0) 
```

In the sentiment analysis section, we'll just need half of the language model - the _encoder_ , so we save that part.

```
 learner.save_encoder('adam3_20_enc') 
```

```
 learner.load_encoder('adam3_20_enc') 
```

Language modeling accuracy is generally measured using the metric _perplexity_ , which is simply `exp()` of the loss function we used.

```
 math.exp(4.165) 
```

```
 64.3926824434624 
```

```
 pickle.dump(TEXT, open(f' {PATH} models/TEXT.pkl','wb')) 
```

#### Testing [ [02:04:53](https://youtu.be/gbceqO8PpBg%3Ft%3D2h4m53s) ]

We can play around with our language model a bit to check it seems to be working OK. First, let's create a short bit of text to 'prime' a set of predictions. We'll use our torchtext field to numericalize it so we can feed it to our language model.

```
 m=learner.model  ss=""". So, it wasn't quite was I was expecting, but I really liked it anyway! The best""" 
```

```
 s = [spacy_tok(ss)]  t=TEXT.numericalize(s)  ' '.join(s[0]) 
```

```
 ". So , it was n't quite was I was expecting , but I really liked it anyway ! The best" 
```

We haven't yet added methods to make it easy to test a language model, so we'll need to manually go through the steps.

```
 # Set batch size to 1  m[0].bs=1  # Turn off dropout  m.eval()  # Reset hidden state  m.reset()  # Get predictions from model  res,*_ = m(t)  # Put the batch size back to what it was  m[0].bs=bs 
```

Let's see what the top 10 predictions were for the next word after our short text:

```
 nexts = torch.topk(res[-1], 10)[1]  [TEXT.vocab.itos[o] for o in to_np(nexts)] 
```

```
 ['film', 
 'movie', 
 'of', 
 'thing', 
 'part', 
 '<unk>', 
 'performance', 
 'scene', 
 ',', 
 'actor'] 
```

…and let's see if our model can generate a bit more text all by itself!

```
 print(ss," \n ")  for i in range(50):  n=res[-1].topk(2)[1]  n = n[1] if n.data[0]==0 else n[0]  print(TEXT.vocab.itos[n.data[0]], end=' ')  res,*_ = m(n[0].unsqueeze(0))  print('...') 
```

```
 _._ So, it wasn't quite was I was expecting, but I really liked it anyway! The best 
```

```
 film ever ! <eos> i saw this movie at the toronto international film festival . i was very impressed . i was very impressed with the acting . i was very impressed with the acting . i was surprised to see that the actors were not in the movie . _..._ 
```

#### Sentiment [ [02:05:09](https://youtu.be/gbceqO8PpBg%3Ft%3D2h5m9s) ]

So we had pre-trained a language model and now we want to fine-tune it to do sentiment classification.

To use a pre-trained model, we will need to the saved vocab from the language model, since we need to ensure the same words map to the same IDs.

```
 TEXT = pickle.load(open(f' {PATH} models/TEXT.pkl','rb')) 
```

`sequential=False` tells torchtext that a text field should be tokenized (in this case, we just want to store the 'positive' or 'negative' single label).

```
 IMDB_LABEL = data.Field(sequential= False ) 
```

This time, we need to not treat the whole thing as one big piece of text but every review is separate because each one has a different sentiment attached to it.

`splits` is a torchtext method that creates train, test, and validation sets. The IMDB dataset is built into torchtext, so we can take advantage of that. Take a look at `lang_model-arxiv.ipynb` to see how to define your own fastai/torchtext datasets.

```
 splits = torchtext.datasets.IMDB.splits(TEXT, IMDB_LABEL, 'data/') 
```

```
 t = splits[0].examples[0] 
```

```
 t.label, ' '.join(t.text[:16]) 
```

```
 ('pos', 'ashanti is a very 70s sort of film ( 1979 , to be precise ) .') 
```

fastai can create a `ModelData` object directly from torchtext `splits` .

```
 md2 = TextData.from_splits(PATH, splits, bs) 
```

Now you can go ahead and call `get_model` that gets us our learner. Then we can load into it the pre-trained language model ( `load_encoder` ).

```
 m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh,  n_layers=nl, dropout=0.1, dropouti=0.4,  wdrop=0.5, dropoute=0.05, dropouth=0.3) 
```

```
 m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1) 
```

```
 m3\. load_encoder (f'adam3_20_enc') 
```

Because we're fine-tuning a pretrained model, we'll use differential learning rates, and also increase the max gradient for clipping, to allow the SGDR to work better.

```
 m3.clip=25\.  lrs=np.array([1e-4,1e-3,1e-2]) 
```

```
 m3.freeze_to(-1)  m3.fit(lrs/2, 1, metrics=[accuracy])  m3.unfreeze()  m3.fit(lrs, 1, metrics=[accuracy], cycle_len=1) 
```

```
 [ 0\. 0.45074 0.28424 0.88458] 
```

```
 [ 0\. 0.29202 0.19023 0.92768] 
```

We make sure all except the last layer is frozen. Then we train a bit, unfreeze it, train it a bit. The nice thing is once you have got a pre-trained language model, it actually trains really fast.

```
 m3.fit(lrs, 7, metrics=[accuracy], cycle_len=2,  cycle_save_name='imdb2') 
```

```
 [ 0\. 0.29053 0.18292 0.93241]  [ 1\. 0.24058 0.18233 0.93313]  [ 2\. 0.24244 0.17261 0.93714]  [ 3\. 0.21166 0.17143 0.93866]  [ 4\. 0.2062 0.17143 0.94042]  [ 5\. 0.18951 0.16591 0.94083]  [ 6\. 0.20527 0.16631 0.9393 ]  [ 7\. 0.17372 0.16162 0.94159]  [ 8\. 0.17434 0.17213 0.94063]  [ 9\. 0.16285 0.16073 0.94311]  [ 10\. 0.16327 0.17851 0.93998]  [ 11\. 0.15795 0.16042 0.94267]  [ 12\. 0.1602 0.16015 0.94199]  [ 13\. 0.15503 0.1624 0.94171] 
```

```
 m3.load_cycle('imdb2', 4) 
```

```
 accuracy(*m3.predict_with_targs()) 
```

```
 0.94310897435897434 
```

A recent paper from Bradbury et al, [Learned in translation: contextualized word vectors](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors) , has a handy summary of the latest academic research in solving this IMDB sentiment analysis problem. Many of the latest algorithms shown are tuned for this specific problem.

![](../img/1_PotEPJjvS-R4C5OCMbw7Vw.png)

As you see, we just got a new state of the art result in sentiment analysis, decreasing the error from 5.9% to 5.5%! You should be able to get similarly world-class results on other NLP classification problems using the same basic steps.

There are many opportunities to further improve this, although we won't be able to get to them until part 2 of this course.

*   For example we could start training language models that look at lots of medical journals and then we could make a downloadable medical language model that then anybody could use to fine-tune on a prostate cancer subset of medical literature.
*   We could also combine this with pre-trained word vectors
*   We could have pre-trained a Wikipedia corpus language model and then fine-tuned it into an IMDB language model, and then fine-tune that into an IMDB sentiment analysis model and we would have gotten something better than this.

There is a really fantastic researcher called Sebastian Ruder who is the only NLP researcher who has been really writing a lot about pre-training, fine-tuning, and transfer learning in NLP. Jeremy was asking him why this is not happening more, and his view was it is because there is not a software to make it easy. Hopefully Fast.ai will change that.

#### Collaborative Filtering Introduction [ [02:11:38](https://youtu.be/gbceqO8PpBg%3Ft%3D2h11m38s) ]

[笔记本](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)

Data available from [http://files.grouplens.org/datasets/movielens/ml-latest-small.zip](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

```
 path='data/ml-latest-small/' 
```

```
 ratings = pd.read_csv(path+'ratings.csv')  ratings.head() 
```

The dataset looks like this:

![](../img/1_Ev47i52AF-qIRHtYTOYm2Q.png)

It contains ratings by users. Our goal will be for some user-movie combination we have not seen before, we have to predict a rating.

```
 movies = pd.read_csv(path+'movies.csv')  movies.head() 
```

![](../img/1_cl9JWMSKPsrYf4hHsxNq-Q.png)

To make it more interesting, we will also actually download a list of movies so that we can interpret what is actually in these embedding matrices.

```
 g=ratings.groupby('userId')['rating'].count()  topUsers=g.sort_values(ascending=False)[:15] 
```

```
 g=ratings.groupby('movieId')['rating'].count()  topMovies=g.sort_values(ascending=False)[:15] 
```

```
 top_r = ratings.join(topUsers, rsuffix='_r', how='inner',  on='userId')  top_r = top_r.join(topMovies, rsuffix='_r', how='inner',  on='movieId') 
```

```
 pd.crosstab(top_r.userId, top_r.movieId, top_r.rating,  aggfunc=np.sum) 
```

![](../img/1_f50pUlwGbsu85fVI-n9-MA.png)

This is what we are creating — this kind of cross tab of users by movies.

Feel free to look ahead and you will find that most of the steps are familiar to you already.

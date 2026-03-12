# 第02讲：统计语言模型

**DATA130030.01，自然语言处理**（阅读时长约60分钟）  
**日期：** 2025年9月17日  
**主讲人：** 周宝健  
**单位：** 复旦大学数据科学学院

---

> **译注（勘误）：** 原文开头有一处孤立的 `$$x$$`，系排版残留，已删除。原文 bigram 推导中将 $v_0$ 误称为 "EOS 起始标记"，应为 BOS，译文已更正。unigram 推导中对 $\lambda$ 求偏导的求和上限原文写为 $|\mathcal{V}|$，应为 $|\mathcal{V}|+1$（含 EOS），译文已更正。

本讲介绍 $N$-gram 语言模型。我们将定义 $N$-gram 语言模型，并说明如何使用最大似然估计（MLE）来学习模型参数；接着讨论处理零概率问题的技术（平滑），并定义困惑度（perplexity）。

**符号说明**  
经过分词后，一个句子表示为 $[w_1, w_2, \ldots, w_t]$，记作 $s = w_{1:t}$。例如，句子 *"the cat saw the dog"* 对应 $s = w_{1:5} = [\textit{the, cat, saw, the, dog}]$。设 $P(X)$ 表示离散随机变量 $X$ 的概率分布，用 $w_i$ 表示句子第 $i$ 个位置的随机变量 $X_i$，即 $P(X_i = w_i) \equiv P(w_i)$。设 $X_{i+1}$ 表示选择 $w_{i+1}$，$X_i$ 表示在 $w_{i+1}$ 之前出现的 $w_i$，则 $P(X_i, X_{i+1})$ 为 $w_i w_{i+1}$ 的联合概率，记作 $P(X_i = w_i, X_{i+1} = w_{i+1}) \equiv P(w_{i:i+1})$。我们定义词表 $\mathcal{V} = \{v_1, v_2, \ldots, v_{|\mathcal{V}|}\}$，以及两个特殊符号 $v_0 = \text{BOS}$（句首）和 $v_{|\mathcal{V}|+1} = \text{EOS}$（句尾）。

---

## 动机

假设你正在做一个语音识别任务，需要设计一个语音识别器。给定一段声学信号，识别器产生两个候选句子：$s_1 =$ *"It's hard to recognize speech"*（语音识别很难）和 $s_2 =$ *"It's hard to wreck a nice beach"*（毁掉一片好海滩很难）。如果模型足够好，它应使 $P(s_1) > P(s_2)$，因为两者的声学信号非常相似，但 $s_1$ 在日常语言中更为合理。因此，语音识别器需要能够为不同的句子序列赋予概率，以比较 $s_1$ 和 $s_2$ 在现实中出现的可能性。**语言建模的任务就是为句子赋予概率**，即我们希望学到一个分布 $P$，使得 $P(s_1) > P(s_2)$。

**我们的任务：** 目标是在给定某语言词表 $\mathcal{V}$ 下的**训练语料库**（一组句子）上学习一个语言模型。利用训练语料库学习模型后，使用**验证语料库**调整模型超参数，最后用**测试语料库**评估训练好的语言模型的性能。

---

## 统计语言模型与 $N$-gram

统计语言模型定义了在给定词表 $\mathcal{V}$ 上句子的概率分布。英语语言模型的词表可能是 $\mathcal{V} = \{\textit{the}, \textit{dog}, \textit{laughs}, \textit{saw}, \ldots\}$，其中 $\mathcal{V}$ 是有限集。语言中的**句子**是词的序列

$$s = w_1 w_2 \ldots w_t \equiv w_{1:t},$$

其中整数 $t \geq 1$ 不固定，$w_i \in \mathcal{V}$（$i \in \{1, \ldots, t-1\}$），并假设 $w_t = \text{EOS}$ 是不在 $\mathcal{V}$ 中的特殊符号。定义 $\mathcal{V}^{\dagger}$ 为词表 $\mathcal{V}$ 上所有句子的集合，它是无限集（句子长度任意）。例如：

$$\mathcal{V}^{\dagger} = \{\text{the dog barks EOS},\ \text{the EOS},\ \text{cat cat cat EOS},\ \text{EOS},\ \ldots\}.$$

语言模型计算 $w_{1:t}$ 的概率 $p(w_{1:t})$，其正式定义如下：

> **定义（语言模型，LM）**  
> 给定有限词表 $\mathcal{V}$，对任意 $t \geq 1$，长度为 $t$ 的句子是序列 $w_{1:t}$，其中 $w_i \in \mathcal{V}$（$i = 1, \ldots, t-1$），$w_t = \text{EOS}$。设 $\mathcal{V}^{\dagger}$ 为所有此类句子的集合。**语言模型**是定义在 $\mathcal{V}^{\dagger}$ 上的概率分布函数 $p(w_{1:t})$，满足：
>
> - 对任意 $w_{1:t} \in \mathcal{V}^{\dagger}$，$p(w_{1:t}) \geq 0$；
> - $\sum_{w_{1:t} \in \mathcal{V}^{\dagger}} p(w_{1:t}) = 1$。

由此，$p(w_{1:t})$ 是定义在 $\mathcal{V}^{\dagger}$ 上的概率分布。语言模型使我们能够**理解**（通过 $p$ 比较任意两个句子的概率）和**生成**（用分布 $p$ 生成句子）。利用链式法则，可将其分解为条件概率之积：

$$p(w_{1:t}) = \prod_{i=1}^t p(w_i \mid w_{1:i-1}),$$

其中 $p(w_1 \mid w_{1:0} = \text{BOS}) = p(w_1)$，$w_{1:i-1}$ 称为 $w_i$ 的**历史**。精确计算 $p(w_{1:t})$ 通常极为困难。例如，仅对5个词的句子 $p(w_{1:5})$，在 $|\mathcal{V}| = 10^4$ 时就需要约 $10^{20}$ 个参数。目前所有语言模型本质上都是对 $p(w_{1:t})$ 的**估计**，有三大类：

1. **统计语言模型**：使用统计方法，基于词序列的概率预测下一个词，即 $N$-gram 模型。
2. **神经语言模型**：基于神经网络，包括两层神经网络（word2vec）和 RNN（如 LSTM、GRU）等架构。
3. **大语言模型（LLM）**：使用自注意力机制（Transformer）的神经网络模型，如 BERT 和 GPT 系列。

**为何需要 $w_t = \text{EOS}$？** 句子长度 $t$ 本身是随机变量。最常见的处理方式是令最后一个词 $w_t$ 始终等于特殊符号 EOS，且该符号只能出现在序列末尾。

近似 $p(w_{1:t})$ 是构建实用语言模型的关键。在**一阶马尔可夫假设**下，$p(w_{1:t}) = \prod_{i=1}^t P(X_i = w_i \mid X_{i-1} = w_{i-1})$。类似地，在**二阶马尔可夫假设**下：

$$p(w_{1:t}) = \prod_{i=1}^t P(X_i = w_i \mid X_{i-2} = w_{i-2},\ X_{i-1} = w_{i-1}),$$

生成句子的步骤如下：

- **步骤1：** 初始化 $i = 1$，令 $x_0 = x_{-1} = \text{BOS}$；
- **步骤2：** 从分布 $p(X_i = x_i \mid X_{i-2:i-1} = x_{i-2:i-1})$ 中采样 $x_i$；
- **步骤3：** 若 $x_i = \text{EOS}$，则返回 $x_{1:i}$；否则令 $i = i + 1$，返回步骤2。

$N$-gram 模型的正式定义如下：

> **定义（$N$-gram 语言模型）**  
> 给定有限词表 $\mathcal{V}$，$N$-gram 语言模型在 $(N-1)$ 阶马尔可夫假设下计算句子 $w_{1:t}$ 的概率：
>
> $$p(w_{1:t}) = \prod_{i=1}^t q(w_i \mid w_{i-N+1:i-1}),$$
>
> 其中 $q(x_N \mid x_{1:N-1})$ 是 $N$-gram $x_{1:N}$ 的条件概率分布，$x_i \in \mathcal{V} \cup \{\text{BOS}\}$（$i = 1, \ldots, N-1$），$x_N \in \mathcal{V} \cup \{\text{EOS}\}$。默认规定 $i \leq 0$ 时 $w_i = \text{BOS}$。

具体而言：

- $N = 1$：**一元模型（Unigram）**，$p(w_{1:t}) = \prod_{i=1}^t q(w_i)$；
- $N = 2$：**二元模型（Bigram）**，$p(w_{1:t}) = \prod_{i=1}^t q(w_i \mid w_{i-1})$；
- $N = 3$：**三元模型（Trigram）**，$p(w_{1:t}) = \prod_{i=1}^t q(w_i \mid w_{i-2:i-1})$。

模型参数约有 $|\mathcal{V}|^N$ 个。以 $|\mathcal{V}| = 10{,}000$、$N = 3$ 为例，参数数量约 $10^{12}$，数量极为庞大。

---

## 从训练语料库通过最大似然估计学习 $q(x_N \mid x_{1:N-1})$

训练数据集为：

$$\mathcal{D}_{\text{tr}} = \{s_1, s_2, \ldots, s_T\} \equiv s_{1:T}, \quad s_i = w_{1:t_i}^{(i)} \in \mathcal{V}^{\dagger}.$$

设模型由参数向量 $\bm{\theta} = [\theta_1, \theta_2, \ldots, \theta_d]^\top$ 刻画。**最大似然估计（MLE）** 的目标是找到使观测数据联合概率最大的参数：

$$\hat{\bm{\theta}} \in \argmax_{\bm{\theta} \in \bm{\Theta}} \left\{ \mathcal{L}_T(\bm{\theta}; \mathcal{D}_{\text{tr}}) := P(s_{1:T}; \bm{\theta}) \right\}.$$

### 一元模型的 MLE

在一元模型假设下：

$$P(s_{1:T}) = \prod_{i=1}^{|\mathcal{V}|+1} \theta_i^{C(v_i)},$$

其中 $C(v_i)$ 为 $v_i$ 在训练语料中出现的次数，参数向量 $\bm{\theta} = [\theta_1, \ldots, \theta_{|\mathcal{V}|}, \theta_{\text{EOS}}]^\top$，约束为 $\sum_{i=1}^{|\mathcal{V}|+1} \theta_i = 1$。

对数似然为：

$$\argmax_{\sum_j \theta_j = 1} \sum_{i=1}^{|\mathcal{V}|+1} C(v_i) \log \theta_i.$$

引入拉格朗日乘子 $\lambda$，构造拉格朗日函数：

$$G(\bm{\theta}, \lambda) = \sum_{i=1}^{|\mathcal{V}|+1} C(v_i) \log \theta_i - \lambda\left(\sum_{i=1}^{|\mathcal{V}|+1} \theta_i - 1\right).$$

令偏导为零（**勘误：** 原文求和上限为 $|\mathcal{V}|$，应为 $|\mathcal{V}|+1$）：

$$\frac{\partial G}{\partial \theta_i} = \frac{C(v_i)}{\theta_i} - \lambda = 0, \quad \frac{\partial G}{\partial \lambda} = \sum_{i=1}^{|\mathcal{V}|+1} \theta_i - 1 = 0.$$

解得：

$$\theta_i^* = \frac{C(v_i)}{\sum_{i=1}^T |s_i|}.$$

即一元模型的 MLE 参数就是**频率估计**。

### 二元模型的 MLE

对于二元模型，令 $\theta_{i,j+1} := q(v_{j+1} \mid v_i)$，约束为对所有 $i$：$\sum_{j=0}^{|\mathcal{V}|} \theta_{i,j+1} = 1$。

对数似然为：

$$\log P(s_{1:T}; \bm{\Theta}) = \sum_{i=0}^{|\mathcal{V}|} \sum_{j=0}^{|\mathcal{V}|} C(v_i, v_{j+1}) \log q(v_{j+1} \mid v_i).$$

类似地引入拉格朗日乘子 $\lambda_i$，令梯度为零，解得：

$$\theta_{i,j+1}^* = q(v_{j+1} \mid v_i) = \frac{C(v_i, v_{j+1})}{\sum_{j=0}^{|\mathcal{V}|} C(v_i, v_{j+1})}.$$

对三元模型及更高阶，MLE 估计为：

$$q(x_N \mid x_{1:N-1}) = \frac{C(x_{1:N})}{C(x_{1:N-1})}.$$

---

## $N$-gram 语言模型的平滑

$N$-gram 模型通常存在**数据稀疏**问题。例如，在3800万词的新闻语料上观察所有三元组后，同来源新文章中仍有约三分之一的三元组是全新未见的。对未见 $N$-gram 直接用 MLE 会导致零概率，因此需要平滑技术——从高频事件中"削减"一些概率质量，分配给未见事件。

### 加法平滑（Additive Smoothing）

对每个 $N$-gram 计数加上一个小量 $\delta$（通常 $0 < \delta \leq 1$）：

$$q_{\text{Add}}(x_N \mid x_{1:N-1}) = \frac{\delta + C(x_{1:N})}{\delta(|\mathcal{V}|+1) + \sum_{x_N \in \mathcal{V} \cup \{\text{EOS}\}} C(x_{1:N})}.$$

最简单的情形是 $\delta = 1$（加一平滑）。

### 线性插值（Linear Interpolation）

以三元模型为例，将三元、二元、一元估计加权混合：

$$q_{\text{Int}}(x_3 \mid x_{1:2}) = \lambda_1 \cdot q_{\text{ML}}(x_3) + \lambda_2 \cdot q_{\text{ML}}(x_3 \mid x_2) + \lambda_3 \cdot q_{\text{ML}}(x_3 \mid x_{1:2}),$$

其中 $\sum_{i=1}^3 \lambda_i = 1$，各 $\lambda_i$ 从验证集学习。一种简化的"分桶"方法（Bucketing）定义为：

$$\lambda_1 = \frac{C(x_{1:2})}{C(x_{1:2}) + \gamma}, \quad \lambda_2 = (1 - \lambda_1) \cdot \frac{C(x_2)}{C(x_2) + \gamma}, \quad \lambda_3 = 1 - \lambda_1 - \lambda_2,$$

其中 $\gamma > 0$ 为唯一超参数。可验证各 $\lambda_i$ 为正且和为1。

### Katz 回退平滑（Katz Back-off）

对二元模型，对任意 $x_1$ 定义：

$$\mathcal{A}(x_1) = \{x_2: C(x_1 x_2) > 0\}, \quad \mathcal{B}(x_1) = \{x_2: C(x_1 x_2) = 0\}.$$

估计为：

$$q_{\text{Katz}}(x_2 \mid x_1) = \begin{cases} \dfrac{C^*(x_1, x_2)}{C(x_1)} & \text{若 } x_2 \in \mathcal{A}(x_1) \\ \alpha(x_1) \cdot \dfrac{q_{\text{ML}}(x_2)}{\sum_{x_2 \in \mathcal{B}(x_1)} q_{\text{ML}}(x_2)} & \text{若 } x_2 \in \mathcal{B}(x_1) \end{cases}$$

其中折扣计数 $C^*(x_1, x_2) = C(x_1, x_2) - \beta$，典型取值 $\beta = 0.5$。对三元模型，类似地定义 $\mathcal{A}(u,v)$ 和 $\mathcal{B}(u,v)$：

$$q_{\text{Katz}}(w \mid u, v) = \begin{cases} \dfrac{C^*(u,v,w)}{C(u,v)} & \text{若 } w \in \mathcal{A}(u,v) \\ \alpha(u,v) \cdot \dfrac{q_{\text{Katz}}(w \mid v)}{\sum_{w \in \mathcal{B}(u,v)} q_{\text{Katz}}(w \mid v)} & \text{若 } w \in \mathcal{B}(u,v) \end{cases}$$

其中 $\alpha(u,v) = 1 - \sum_{w \in \mathcal{A}(u,v)} \frac{C^*(u,v,w)}{C(u,v)}$。

### Kneser-Ney 平滑（Kneser-Ney Smoothing）

给定折扣参数 $\delta \in [0,1]$，二元 KN 平滑为：

$$q_{\text{KN}}(x_2 \mid x_1) = \frac{\max\{C(x_1 x_2) - \delta,\ 0\}}{\sum_{x \in \mathcal{V} \cup \{\text{EOS}\}} C(x_1 x)} + \lambda_{x_1} \cdot q_{\text{KN}}(x_2),$$

其中一元概率衡量词 $x_2$ 出现在新上下文中的可能性：

$$q_{\text{KN}}(x_2) = \frac{|\{x: C(x x_2) > 0\}|}{|\{(x_1, x_2): C(x_1 x_2) > 0\}|},$$

参数 $\lambda_{x_1}$ 为：

$$\lambda_{x_1} = \frac{\delta \cdot |\{w: C(x_1 w) > 0\}|}{\sum_v C(x_1 v)}.$$

推广到 $N$-gram：

$$q_{\text{KN}}(x_N \mid x_{1:N-1}) = \frac{\max\{C(x_{1:N}) - \delta,\ 0\}}{\sum_{x \in \mathcal{V} \cup \{\text{EOS}\}} C(x_{1:N-1} x)} + \lambda_{x_{1:N-1}} \cdot q_{\text{KN}}(x_N \mid x_{2:N-1}),$$

其中：

$$\lambda_{x_{1:N-1}} = \frac{\delta \cdot |\{x: C(x_{1:N-1} x) > 0\}|}{\sum_{x \in \mathcal{V} \cup \{\text{EOS}\}} C(x_{1:N-1} x)}.$$

改进版 Kneser-Ney 平滑已在 [kenlm](https://github.com/kpu/kenlm) 中实现。

### 愚蠢回退（Stupid Back-off）

不带折扣的回退（非真正的概率）：

$$S(x_N \mid x_{1:N-1}) = \begin{cases} \dfrac{C(x_{1:N})}{C(x_{1:N-1})} & \text{若 } C(x_{1:N}) > 0 \\ 0.4 \cdot S(x_N \mid x_{2:N-1}) & \text{否则} \end{cases}$$

$$S(x_N) = \frac{C(x_N)}{T},$$

其中 $T$ 为训练语料的总大小。

---

## 用困惑度评估 $N$-gram

设测试数据为句子 $s_1, s_2, \ldots, s_m$，一个好的模型应对这些真实句子赋予较高概率。总体质量可用：

$$\prod_{i=1}^m p(s_i)$$

来衡量。设 $M = \sum_{i=1}^m t_i$ 为测试语料的总词数，**平均对数概率**定义为：

$$\frac{1}{\sum_{i=1}^m |s_i|} \log_2 \prod_{i=1}^m p(s_i) = \frac{1}{\sum_{i=1}^m |s_i|} \sum_{i=1}^m \log_2 p(s_i).$$

**困惑度（Perplexity）** 定义为：

$$\operatorname{PPL}(s_{1:m}) := 2^{-l}, \quad \text{其中 } l = \frac{1}{\sum_{i=1}^m |s_i|} \sum_{i=1}^m \log_2 p(s_i).$$

困惑度是正数，**值越小，语言模型对未见数据的建模能力越强**。其直觉含义为：*"如果在每一步按照语言模型的概率分布随机采样词，平均需要采样多少次才能得到正确的词？"*（Neubig, 2017, 第9页）。

**示例：** 若一元模型对所有词赋予相同概率，则 $\operatorname{PPL}(s_{1:m}) = |\mathcal{V}|$——平均需要从全部词表中选择。若 $|\mathcal{V}| = 1{,}000{,}000$ 且一个好的语言模型的困惑度为30，则平均只需选约30个词。

### 熵与交叉熵

困惑度本质上衡量语言模型的**熵**。给定离散随机变量 $X$，**熵**定义为：

$$H(p) \triangleq -\sum_{i=1}^n p(x_i) \log p(x_i).$$

分布 $q$ 相对于分布 $p$ 的**交叉熵**定义为：

$$H(p, q) = -\sum_{x_i \in \mathcal{X}} p(x_i) \log q(x_i).$$

在语言模型中，$p$ 为真实分布，$q$ 为从训练语料估计的分布。由于真实分布未知，我们无法直接计算 $H(p)$ 或 $H(p,q)$。但可以利用：

$$H(p, q) = H(p) + \underbrace{\sum_{x_i \in \mathcal{X}} p(x_i) \left(\log p(x_i) - \log q(x_i)\right)}_{D_{\text{KL}}(p \| q) \geq 0} \geq H(p).$$

因此 $H(p,q)$ 是 $H(p)$ 的上界，$q$ 越接近 $p$，$H(p,q)$ 越接近 $H(p)$（等号当且仅当 $p = q$ 时成立）。

由 **Shannon-McMillan-Breiman 定理**（Jurafsky, 2022, 公式3.49）：

$$H(p) \leq H(p, q) = \lim_{n \to \infty} -\frac{1}{n} \log q(w_{1:n}) \approx H(s_{1:m}) \triangleq -\frac{1}{|s_{1:m}|} \log p(s_{1:m}).$$

**困惑度**正是该近似量 $H(s_{1:m})$ 的指数形式：

$$\operatorname{PPL}(s_{1:m}) = 2^{-\frac{1}{|s_{1:m}|} \log_2 p(s_{1:m})} = 2^{H(s_{1:m})}.$$

采用指数形式没有特别原因，但它使数值差异更为显著，比 $H(W)$ 对模型性能更为敏感。

**困惑度的优势：**
1. 比熵值更直观易记（如100–200的困惑度范围，比6.64–7.64比特的熵值更具可解释性）；
2. 10%的困惑度改善比2%的熵减小更具感知显著性；
3. 可直接用留出集或测试数据计算，困惑度越低说明模型越接近生成观测数据的"真实"模型。

$N$-gram 模型的核心组件曾被用于第一代谷歌翻译（Brants et al., 2007）。然而，作为纯统计语言模型，它已先后被基于 RNN 的模型和大语言模型（LLM）所取代。

---

## 参考文献

- https://cs229.stanford.edu/lectures-spring2023/cs229-probability_review.pdf
- https://cs229.stanford.edu/notes2022fall/main_notes.pdf
- https://anoopsarkar.github.io/nlp-class/assets/slides/prob.pdf
- https://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/lm.pdf
- https://pages.cs.wisc.edu/~jerryzhu/cs838/LM.pdf
- https://www3.nd.edu/~dchiang/teaching/nlp/2021/notes/chapter2v2.pdf
- https://www.fit.vut.cz/study/phd-thesis-file/283/283.pdf

---

## 附录

**Jensen 不等式：** 对凹函数 $\varphi: \mathbb{R} \to \mathbb{R}$ 及正权 $a_i$：

$$\varphi\!\left(\sum_{i=1}^n \frac{x_i}{n}\right) \geq \frac{1}{n} \sum_{i=1}^n \varphi(x_i) \quad (\text{当 } a_i = 1).$$

> **定理（最大熵）**  
> 对随机变量 $X$ 及其概率分布 $p$，当 $p$ 为均匀分布时，$X$ 的熵最大。

**证明：**  
令 $\varphi(y) = -y \log y$（规定 $\varphi(0) = 0$），设 $X$ 有 $n$ 个可能取值 $x_1, \ldots, x_n$，概率为 $p(x_1), \ldots, p(x_n)$。由于 $\varphi(y)$ 在 $[0,1]$ 上为凹函数，由 Jensen 不等式：

$$\begin{aligned}
n \cdot \varphi\!\left(\sum_{i=1}^n \frac{y_i}{n}\right) &\geq \sum_{i=1}^n \varphi(y_i) \\
n \cdot \left(-\frac{1}{n}\right) \log\!\left(\frac{1}{n}\right) &\geq -\sum_{i=1}^n p(x_i) \log p(x_i) := H(p),
\end{aligned}$$

即 $\log(n) \geq H(p)$。上界在 $p(x_i) = 1/n$（均匀分布）时取到，证毕。

更多内容见 Conrad (2004)。

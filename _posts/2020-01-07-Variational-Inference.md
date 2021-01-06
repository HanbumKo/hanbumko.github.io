---
title: Variational Inference
updated: 2021-01-07 00:33
---



# Variational Inference

랩 내의 스터디중 하나에서 [Kingma](http://www.dpkingma.com/)의 Thesis를 공부하고 있는데, 정리도 할겸 복습용으로 작성중인 글입니다. [Full Thesis](https://pure.uva.nl/ws/files/17891313/Thesis.pdf)

##### contents
1. [Introduction and Background](#introduction-and-background)
2. Variational Autoencoders
3. Semi-Supervised Learning
4. Deeper Generative Models
5. Inverse Autoregressive Flow
6. Variational Dropout and Local Reparameterization
7. ADAM: A Method for Stochastic Optimization
8. Conclusion



<div class="divider"></div>


## Introduction and Background

$$
prior\; distribution  \overset { seeing\; data }{ \rightarrow  } posterior\; distribution
$$

위를 수행하는 방법중 하나가 **variational inference**이다.

관심있는 변수들의 벡터를 lower case $$\mathbf{x}$$로 나타낸다.

<br>

관측된 $$\mathbf{x}$$는 알려지지 않은 process (unknown underlying process)에 의하여 random sample 된것이라 가정한다.

true distribution은 $$p^\ast (\mathbf{x})$$로 표기한다.

<br>

we attempt to approximate this underlying process with a chosen model $$p_\theta (\mathbf{x})$$, with parameter $$\theta$$

$$
\mathbf{x} \quad \sim \quad p_\theta (\mathbf{x}) \tag{1.1} \label{eq:1_1}
$$


<br>

이 true distribution $$p^\ast (\mathbf{x})$$의 파라미터를 추론하는 과정을 **Learning**이라 한다.

$$
p_\theta (\mathbf{x}) \approx p^\ast (\mathbf{x}) \tag{1.2} \label{eq:1_2}
$$

$$p_\theta (\mathbf{x})$$는 충분히 flexible 하고 충분히 정확하게 모델링 될 수 있어야 한다.

<br>

### Conditional Models

classification 혹은 regression 문제들은 $$p_\theta(\mathbf{x})$$가 아닌 conditional model $$p_\theta(\mathbf{y} \mid \mathbf{x})$$에 관심이 있다.

conditional model $$p_\theta(\mathbf{y} \mid \mathbf{x})$$는 underlying conditional distribution $$p^\ast (\mathbf{y} \mid \mathbf{x})$$를 approximate 한다.

$$p^\ast (\mathbf{y} \mid \mathbf{x})$$: a distribution over the value of variable $$\mathbf{y}$$, conditioned on variable $$\mathbf{x}$$

$$\mathbf{x}$$를 model의 **input**이라 한다.

이 또한 unconditional의 경우처럼 $$p_\theta(\mathbf{y} \mid \mathbf{x})$$가 $$p^\ast (\mathbf{y} \mid \mathbf{x})$$와 충분히 가깝게 최적화 되어야 한다.

$$
\begin{align}
p_\theta(\mathbf{y} \mid \mathbf{x}) \approx p^\ast (\mathbf{y} \mid \mathbf{x}) \tag{1.3} \label{eq:1_3}
\end{align}
$$

$$\eqref{eq:1_1}$$, $$\eqref{eq:1_2}$$, $$\eqref{eq:1_3}$$






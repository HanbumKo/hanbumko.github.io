---
title: Variational Inference
updated: 2021-01-10 18:21
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
p_\theta (\mathbf{x}) \approx p^\ast (\mathbf{x}) \tag{1_2} \label{eq:1.2}
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

예시)

* image classification
  * $$\mathbf{x}$$: an image
  * $$\mathbf{y}$$: image's class

<br>

notation의 간결성을 위해 대부분 unconditional 이라고 가정하지만 conditional의 경우도 가능하다


### Parameterizing Conditional Distributions with Neural Networks

neural networks(NN)를 conditional probability density를 계산하는 함수라고 볼 수 있다.

NN은 flexible하게 모델링 하는 방법 중 하나이다. Stochastic Gradient Descent(SGD)를 통해 효과적으로 최적화 될 수 있다.

deep neural network를 간단히 $$NeuralNet(\cdot)$$로 표기한다.

image classification의 경우 다음과 같이 나타낼 수 있다.

$$
\begin{align}
\mathbf{p} &= NeuralNet(\mathbf{x}) \tag{1.4} \label{eq:1_4} \\
p_\theta(\mathbf{y} \mid \mathbf{x}) &= Categorical(\mathbf{y} ; \mathbf{p}) \tag{1.5} \label{eq:1_5}
\end{align}
$$


<br>

### Directed Graphical Models and Neural Networks

directed (probabilistic) grahpical models는 모든 변수가 directed acyclic graph로 정리되는 모델이다.

$$
\begin{align}
p_\theta(\mathbf{x}_1, \cdots, \mathbf{x}_\mathcal{M}) = \prod_{j=1}^{\mathcal{M}} {p_\theta(\mathbf{x}_j \mid p_a(\mathbf{x}_j))} \tag{1.6} \label{eq:1_6}
\end{align}
$$

$$p_a(\mathbf{x}_j)$$: node $$\mathbf{x}_j$$의 parent vatiable. root node일 경우는 unconditional로 쓴다.

directed graphical models 혹은 Bayesian networks라고 한다.

$${p_\theta(\mathbf{x}_j \mid p_a(\mathbf{x}_j))} $$는 lookup table 혹은 linear model로 parameterized 될 수 있으며 뉴럴넷은 조금 더 flexible한 방법이다.

Conditional distribution을 parameterize 할 때에는 뉴럴넷이 파라미터 $$\eta$$를 생성한다

$$
\begin{align}
\eta &= NeuralNet(p_a(\mathbf{x})) \tag{1.7} \label{eq:1_7}\\
p_\theta(\mathbf{x} \mid p_a(\mathbf{x})) &= p_\theta(\mathbf{x} \mid \eta) \tag{1.8} \label{eq:1_8}
\end{align}
$$

<br>

### Learning in Fully Observed Models with Neural Nets

#### Dataset
$$
\begin{align}
\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \cdots, \mathbf{x}^{(N)} \} \equiv \{ \mathbf{x}^{(i)} \}_{i=1}^N \equiv \mathbf{x}^{(1:N)} \tag{1.9} \label{eq:1_9}
\end{align}
$$

-> i.i.d.

#### Maximum Likelihood and Minibatch SGD

probabilistic models의 가장 흔한 평가기준(criterion)은 maximum log-likelihood(ML)이다.

log-likelihood criterion을 최대화 하는 것은 Kullback-Leibler divergence를 최소화 하는 것과 같다. (data와 model distribution)

ML criterion을 이용해 log-probabilities의 합 혹은 평균을 최대화하는 파라미터 $$\theta$$를 찾는다.

$$
\begin{align}
\log{p_\theta(\mathcal{D})} = \sum_{\mathbf{x} \in \mathcal{D}}\log{p_\theta(\mathbf{x})} \tag{1.10} \label{eq:1_10}
\end{align}
$$

$$\nabla_\theta \log{p_\theta(\mathcal{D})}$$: batch gradient

-> 데이터 사이즈 $$N_\mathcal{D}$$이 증가함에 따라 시간도 linear 하게 증가

Stochastic Gradient Descent (SGD)

-> $$\mathcal{D}$$에서 랜덤하게 미니배치 뽑아서 하는 것

$$\mathcal{M} \subset \mathcal{D}, \quad size: N_\mathcal{M}$$

미니배치를 이용하면 ML criterion의 불편추정량(unbiased estimator)을 얻을 수 있다.

$$
\frac{1}{N_\mathcal{D}}\log{p_\theta(\mathcal{D})} \simeq \frac{1}{N_\mathcal{M}}\log{p_\theta(\mathcal{M})} = \frac{1}{N_\mathcal{M}} \sum_{\mathbf{x} \in \mathcal{M}}{\log{p_\theta (\mathbf{x}) }} \tag{1.11} \label{eq:1_11}
$$

$$\simeq$$: unbiased estimator

unbiased estimator $$\log_{\theta}{(\mathcal{M})}$$은 미분 가능하고 unbiased stochastic gradients를 얻을 수 있다.

$$
\frac{1}{N_\mathcal{D}} \nabla_\theta \log{p_\theta(\mathcal{D})} \simeq \frac{1}{N_\mathcal{M}} \nabla_\theta \log{p_\theta(\mathcal{M})} = \frac{1}{N_\mathcal{M}} \sum_{\mathbf{x} \in \mathcal{M}}\nabla_\theta{\log{p_\theta (\mathbf{x}) }} \tag{1.12} \label{eq:1_12}
$$

#### Bayesian inference

Bayesian의 관점에서 ML을 maximum a posteriori(MAP) 추정으로 개선할 수 있다. 혹은 full approximate posterior distribution을 추정할 수 있다.


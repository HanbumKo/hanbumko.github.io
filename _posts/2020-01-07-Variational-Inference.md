---
title: Variational Inference
updated: 2021-01-18 19:01
---



# Variational Inference

랩 내의 스터디중 하나에서 [Kingma](http://www.dpkingma.com/)의 Thesis를 공부하고 있는데, 정리도 할겸 복습용으로 작성중인 글입니다. [Full Thesis](https://pure.uva.nl/ws/files/17891313/Thesis.pdf).

연구 결과를 공개해준 Kingma에게 감사합니다.

##### contents
1. [Introduction and Background](#introduction-and-background)
2. [Variational Autoencoders](#variational-autoencoders)
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

$$p_a(\mathbf{x}_j)$$: node $$\mathbf{x}_j$$의 parent variable. root node일 경우는 unconditional로 쓴다.

*directed graphical models* 혹은 *Bayesian networks*라고 한다.

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

log-likelihood criterion을 최대화 하는 것은 data와 model distribution 사이의 Kullback-Leibler divergence를 최소화 하는 것과 같다.

ML criterion을 이용해 log-probabilities의 합 혹은 평균을 최대화하는 파라미터 $$\theta$$를 찾는다.

$$
\begin{align}
\log{p_\theta(\mathcal{D})} = \sum_{\mathbf{x} \in \mathcal{D}}\log{p_\theta(\mathbf{x})} \tag{1.10} \label{eq:1_10}
\end{align}
$$

$$\nabla_\theta \log{p_\theta(\mathcal{D})}$$: batch gradient

-> 데이터 사이즈 $$N_\mathcal{D}$$이 증가함에 따라 시간도 linear 하게 증가

*Stochastic Gradient Descent (SGD)*

-> $$\mathcal{D}$$에서 랜덤하게 미니배치 뽑아서 하는 것

$$\mathcal{M} \subset \mathcal{D}, \quad size: N_\mathcal{M}$$

미니배치를 이용하면 ML criterion의 불편추정량(unbiased estimator)을 얻을 수 있다.

$$
\frac{1}{N_\mathcal{D}}\log{p_\theta(\mathcal{D})} \simeq \frac{1}{N_\mathcal{M}}\log{p_\theta(\mathcal{M})} = \frac{1}{N_\mathcal{M}} \sum_{\mathbf{x} \in \mathcal{M}}{\log{p_\theta (\mathbf{x}) }} \tag{1.11} \label{eq:1_11}
$$

$$\simeq$$: unbiased estimator

unbiased estimator $$\log p_{\theta}{(\mathcal{M})}$$은 미분 가능하고 unbiased stochastic gradients를 얻을 수 있다.

$$
\frac{1}{N_\mathcal{D}} \nabla_\theta \log{p_\theta(\mathcal{D})} \simeq \frac{1}{N_\mathcal{M}} \nabla_\theta \log{p_\theta(\mathcal{M})} = \frac{1}{N_\mathcal{M}} \sum_{\mathbf{x} \in \mathcal{M}}\nabla_\theta{\log{p_\theta (\mathbf{x}) }} \tag{1.12} \label{eq:1_12}
$$

#### Bayesian Inference

Bayesian의 관점에서 ML을 maximum a posteriori(MAP) 추정으로 개선할 수 있다. 혹은 full approximate posterior distribution을 추정할 수 있다.

<br>

### Learning and Inference in Deep Latent Variable Models

#### Latent Variables

latent variables는 보여지지 않은 variables이다. 주로 $$\mathbf{z}$$로 나타낸다.

observed variable $$\mathbf{x}$$에 대해 unconditional 모델링의 경우에 directed grahpical model은 $$\mathbf{x}$$, $$\mathbf{z}$$에 관한 joint distribution $$p_\theta(\mathbf{x}, \mathbf{z})$$로 나타낼 수 있다.

그리고 $$p_\theta(\mathbf{x})$$는 marginal로 나타낼 수 있다.

$$
p_\theta(\mathbf{x}) = \int{p_\theta(\mathbf{x}, \mathbf{z})}d\mathbf{z} \tag{1.13} \label{eq:1_13}
$$

*(single datapoint) marginal likelihood* 혹은 *model evidence*라고 부른다.

만약 $$\mathbf{z}$$가 discrete이고 $$p_\theta(\mathbf{x} \mid \mathbf{z}) \; \sim \mathcal{N}$$이면 $$q_\phi(\mathbf{z} \mid \mathbf{x})$$는 mixture-of-Gaussian-distribution이다.

continuous한 $$\mathbf{z}$$의 경우에는 $$p_\theta(\mathbf{x})$$가 infinite mixture라고 볼 수 있다.

이러한 marginal distributions를 *compound probability distributions* 라고도 부른다.

#### Deep Latent Variable Models

latent variable model $$p_\theta(\mathbf{x}, \mathbf{z})$$를 뉴럴넷으로 parameterize한 경우를 *deep latent variable model(DLVM)*이라고 (사용)한다.

$$p_\theta(\mathbf{x}, \mathbf{z} \mid \mathbf{y})$$ 같이 conditioned로도 표현 가능하다.

DLVMs의 장점 중 하나는, marginal distribution $$p_\theta(\mathbf{x})$$가 매우 복잡해 질 수 있는데, 이 복잡한 underlying distribution $$p^\ast(\mathbf{x})$$를 잘 approximate 할 수 있는 것이다.

$$
p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{z}) p_\theta( \mathbf{x} \mid \mathbf{z} ) \tag{1.14} \label{eq:1_14}
$$

$$p_\theta(\mathbf{z})$$ and/or $$p_\theta(\mathbf{x} \mid \mathbf{z})$$는 정해져 있다.

$$p(\mathbf{z})$$를 보통 $$\mathbf{z}$$에 대한 *prior distribution*이라 한다. (어떠한 observations에도 conditioned 되지 않았기 때문에)

#### Example DLVM for multivariate Bernoulli data

Variational Auto Encoder(VAE)에서 간단한 예제를 살펴 볼 수 있다.

$$
\begin{align}
p(\mathbf{z}) &= \mathcal{N}(\mathbf{z} ; 0, \mathbf{I}) \tag{1.15} \label{eq:1_15} \\
p &= DecoderNeuralNet_\theta(\mathbf{z}) \tag{1.16} \label{eq:1_16} \\
\log {p(\mathbf{x} \mid \mathbf{z})} &= \sum_{j=1}^{D}\log {p(\mathbf{x}_j \mid \mathbf{z})} = \sum_{j=1}^{D} \log {Bernoulli(\mathbf{x}_j ; p_j)} \tag{1.17} \label{eq:1_17} \\
&= \sum_{j=1}^{D} \mathbf{x}_j \log {p_j} + (1-\mathbf{x_j}) \log {(1-p_j)} \tag{1.18} \label{eq:1_18}
\end{align}
$$

where $$\forall p_j \in \mathbf{p} \; : \; 0 \le p_j \le 1$$

, D is dimensionality of $$\mathbf{x}$$

, $$Bernoulli(. ; p)$$ is p.m.f. of Bernoulli distribution

<br>

### Intractabilities

DLVMs에서 maximum likelihood 학습의 주요 어려움은 데이터의 marginal probability가 주로 intractable(다루기 어렵다) 하다는 것이다. $$p_\theta(\mathbf{x}) = \int p_\theta (\mathbf{x}, \mathbf{z})d\mathbf{z}$$가 analytic한 해 또는 추정량을 갖지 못한다.

따라서 파라미터들에 대해 미분과 최적화를 하지 못한다.

$$p_\theta(\mathbf{x})$$의 intractability는 posterior distribution $$p_\theta(\mathbf{z} \mid \mathbf{x})$$와 관련되어 있다.

$$
p_\theta(\mathbf{z} \mid \mathbf{x}) = \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{x})} \tag{1.19} \label{eq:1_19}
$$

$$p_\theta(\mathbf{x}, \mathbf{z})$$는 tractable 할 때,

$$p_\theta(\mathbf{x})$$가 tractable 하면 $$p_\theta( \mathbf{z} \mid \mathbf{x})$$도 tractable 하다.(vice-versa)

마찬가지로 뉴럴넷의 파라미터에 대한 posterior $$p(\theta \mid \mathcal{D})$$도 정확하게 계산되기 힘들고 approximate 추정 방법이 필요하다.

<br>

### Research Questions and Contributions

**Research Question 1:** 큰 데이터셋이 존재할때, DLVMs에서 어떻게 효과적으로 posterior 추정과 ML 추정을 할 것인지?

-> chapter 2에서 *reparameterization trick* 과 함께 다룬다. *variational autoencoder*(VAE)이 뉴럴넷을 이용한 추정 모델과 뉴럴넷 이용한 generative model을 조합해서 사용한다. 이 두 네트워크의 joint optimization하는 간단한 방법을 다룬다.

**Research Question 2:** VAE를 사용해서 최신의 semi-supervised classification 결과들을 개선시킬 수 있을지?

-> chapter 3에서 VAE가 semi-supervised learning을 다루는 방법을 설명한다. normalizing flows가 posterior distributions을 parameterizing 하는 flexible한 방법을 제공하는데, high-dimensional latent spaces에서는 잘 안된다.

**Research Question 3:** high-dimensional latent space를 잘 다루는 practical한 normalizing flow가 존재하는지?

-> chapter 5에서 high-dimensional latent spaces에 대해 highly non-Gaussian posterior distributions의 추정을 하는 *inverse autoregressive flows* 를 다룬다. VAE에서도 어떻게 쓰이나 살펴본다. 

**Research Question 4:** reparameterization-based gradient 추정량을 variance가 미니배치 사이즈와 반비례하게 증가하는 gradient estimator를 사용해  개선시킬 수 있는지?

-> chapter 6에서 Gaussian posterior의 variational inference 효율성을 개선시키기 위한 *local reparameterization trick*을 다룬다. 이 방법은 dropout을 추가적인 (Bayesian) 시각으로 볼 수 있도록 한다.

**Research Question 5:** 현재 존재하는 stochastic gradient-based 최적화 방법들을 개선 시킬 수 있는지?

-> chapter 7에서 *Adam*을 소개한다.


<div class="divider"></div>


## Variational Autoencoders

### Encoder or Approximate Posterior

$$p_\theta(\mathbf{x}, \mathbf{z})$$를 latent-variable 모델이라 하자

DLVMs의 posterior 추론과 학습에서 intractable한 문제를 tractable로 하기 위해 parametric inference model $$q_\phi(\mathbf{z} \mid \mathbf{x})$$를 정의한다. $$q_\phi(\mathbf{z} \mid \mathbf{x})$$를 encoder 라고 한다. $$\phi$$는 inference model의 파라미터이고 *variational parameters*라고도 한다.

$$
q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p_\theta(\mathbf{z} \mid \mathbf{x}) \tag{2.1} \label{eq:2_1}
$$

DLVM 처럼 inference 모델은 어떠한 directed graphical model이 될 수 있다.

$$
q_\phi(\mathbf{z} \mid \mathbf{x}) = q_\phi(\mathbf{z}_1, \cdots, \mathbf{z}_M \mid \mathbf{x}) = \prod_{j=1}^M q_\phi(\mathbf{z}_j \mid Pa(\mathbf{z}_j), \mathbf{x}) \tag{2.2} \label{eq:2_2}
$$

$$q_\phi(\mathbf{z} \mid \mathbf{x})$$는 DNN으로 나타낼 수 있다.

$$
\begin{align}
(\mu, \, log\sigma) &= EncoderNeuralNet_\phi(\mathbf{x}) \tag{2.3} \label{eq:2_3} \\
q_\phi(\mathbf{z} \mid \mathbf{x}) &= \mathcal{N}(\mathbf{z} ; \mu, diag(\sigma)) \tag{2.4} \label{eq:2_4}
\end{align}
$$

모든 데이터에 같은 모델을 쓰는 것을 *amortized variational inference*라고 한다.

<br>

### Evidence Lower Bound (ELBO)

variational autoencoder의 최적화의 목표(objective)는 evidence lower bound (ELBO)이다. *variational lower bound*라고도 한다.

![figure2_1](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/figure2_1.png?raw=true)

variational parameter $$\phi$$를 포함하는 inference model $$q_\phi(\mathbf{z} \mid \mathbf{x})$$에서,

$$
\begin{align}
\log p_\theta(\mathbf{x}) &= \mathbb{E}_{ q_\phi(\mathbf{z} \mid \mathbf{x}) }[\log p_\theta(\mathbf{x})] \tag{2.5} \label{eq:2_5} \\
&= \mathbb{E}_{ q_\phi(\mathbf{z} \mid \mathbf{x}) }[\log [ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z} \mid \mathbf{x})} ]] \tag{2.6} \label{eq:2_6} \\
&= \mathbb{E}_{ q_\phi(\mathbf{z} \mid \mathbf{x}) }[\log [ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})} \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})} ]] \tag{2.7} \label{eq:2_7} \\
&= \underbrace{ \mathbb{E}_{ q_\phi(\mathbf{z} \mid \mathbf{x}) }[\log [ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}]] }_{ \substack{ \mathcal{L}_{\theta,\phi}( \mathbf{x}) \\  \text{(ELBO)}}} + \underbrace{ \mathbb{E}_{ q_\phi(\mathbf{z} \mid \mathbf{x}) }[\log [ \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})}]] }_{ D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \, \| \, p_\theta(\mathbf{z} \mid \mathbf{x}))} \tag{2.8} \label{eq:2_8}
\end{align}
$$

으로 식을 변형할 수 있다. 두번째 항은

$$
D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \, \| \, p_\theta(\mathbf{z} \mid \mathbf{x})) \geq 0 \tag{2.9} \label{eq:2_9}
$$

-> true posterior $$p_\theta( \mathbf{z} \mid \mathbf{x} )$$와 똑같으면 0을 갖는다. 분포가 얼마나 다른지 나타낸다.

$$\eqref{eq:2_8}$$의 첫번째 항은 variational lower bound, evidence lower bound (ELBO) 이다.

$$
\mathcal{L}_{\theta, \phi}(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})] \tag{2.10} \label{eq:2_10}
$$

KL divergence가 음수가 아니기 때문에 ELBO는 데이터의 log-likelihood의 lower bound이다.

$$
\begin{align}
\mathcal{L}_{\theta, \phi}(\mathbf{x}) &= \log p_\theta(\mathbf{x}) - D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})) \tag{2.11} \label{eq:2_11} \\
&\leq \log p_\theta(\mathbf{x}) \tag{2.12} \label{eq:2_12}
\end{align}
$$

KL divergence $$D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x}))$$는 두 가지 '거리'를 나타낸다.

1. approximate posterior와 true posterior의 KL divergence (by definition)
2. ELBO $$\mathcal{L}_{\theta, \phi}(\mathbf{x})$$와 marginal likelihood $$\log p_\theta(\mathbf{x})$$의 차이 -> *tightness*라고도 한다. 더 잘 추론할수록 차이가 줄어든다.

![figure2_2](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/figure2_2.png?raw=true)

#### A Double-Edged Sword

식 $$\eqref{eq:2_11}$$에서, ELBO $$\mathcal{L}_{\theta, \phi}(\mathbf{x})$$의 최대화는 두 가지를 동시에 최적화 한다고 볼 수 있다.

1. approximately maximize the marginal likelihood $$p_\theta(\mathbf{x})$$ -> generative 모델이 최적화
2. minimize KL divergence -> $$q_\phi(\mathbf{z} \mid \mathbf{x})$$가 최적화

<br>

### Stochastic Gradient-Based Optimization of the ELBO

i.i.d. dataset이 주어질때 ELBO의 objective는 ELBO의 합 혹은 평균이다.

$$
\mathcal{L}_{\theta, \phi}(\mathcal{D}) = \sum_{x \in \mathcal{D}} \mathcal{L}_{\theta, \phi}(\mathbf{x}) \tag{2.13} \label{eq:2_13}
$$

각 datapoint ELBO의 gradient $$\nabla_{\theta, \phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$는 일반적으로 intractable 하다. 그러나 좋은 불편추정량 $$\tilde{\nabla}_{\theta, \phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$이 존재하여 minibatch SGD를 할 수 있다.

$$
\begin{align}
\nabla_{\theta} \mathcal{L}_{\theta, \phi}(\mathbf{x}) &= \nabla_{\theta} \mathbb{E}{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})] \tag{2.14} \label{eq:2_14} \\
&= \mathbb{E}{q_\phi(\mathbf{z} \mid \mathbf{x})}[\nabla_{\theta} (\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}))] \tag{2.15} \label{eq:2_15} \\
&\simeq \nabla_{\theta}( \log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}) ) \tag{2.16} \label{eq:2_16} \\
&= \nabla_{\theta}( \log p_\theta(\mathbf{x}, \mathbf{z})) \tag{2.17} \label{eq:2_17}
\end{align}
$$

식 $$\eqref{eq:2_17}$$은 식 $$\eqref{eq:2_15}$$의 Monte Carlo 추정량이다. ($$\mathbf{z}$$는 $$q_\phi(\mathbf{z} \mid \mathbf{x})$$로부터 random sample 됨)

variational parameter $$\phi$$에 관한 불편(unbiased) gradients는 얻기가 어렵다. (ELBO의 expectation이 $$q_\phi(\mathbf{z} \mid \mathbf{x})$$에 대한 것인데 이것이 $$\phi$$의 함수라서)

$$
\begin{align}
\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x}) &= \nabla_{\phi} \mathbb{E}{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})] \tag{2.18} \label{eq:2_18} \\
&\neq \mathbb{E}{q_\phi(\mathbf{z} \mid \mathbf{x})}[\nabla_{\phi} (\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}))] \tag{2.19} \label{eq:2_19}
\end{align}
$$

latent variable이 continuous한 경우 불편추정량 계산을 위해 reparameterization trick을 사용할 수 있다. (discrete한 경우는 앞으로 나올 [Score function estimator](#score-function-estimator)에서 설명한다.)

<br>

### Reparameterization trick

![figure2_3](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/figure2_3.png?raw=true)

![algorithm1](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/algorithm1.png?raw=true)

#### Change of Variables

random variable $$\mathbf{z} \; \sim \; q_\phi(\mathbf{z} \mid \mathbf{x})$$를 다른 random variable $$\epsilon$$를 사용하여 미분가능하게 변환을 할 수 있다.

$$
\mathbf{z} = g(\epsilon, \phi, \mathbf{x}) \tag{2.20} \label{eq:2_20}
$$

random variable $$\epsilon$$은 $$\mathbf{x}$$ 또는 $$\phi$$에 대해 독립이다.

#### Gradient of Expectation under change of variable

이렇게 변환을 하게 되면 expectation을 $$\epsilon$$에 대해 다시 쓸 수 있다.

$$
\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] = \mathbb{E}_{p(\epsilon)}[f(\mathbf{z})] \tag{2.21} \label{eq:2_21}
$$

where $$\mathbf{z} = g(\epsilon, \phi, \mathbf{x})$$

gradient는

$$
\begin{align}
\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] &= \nabla_\phi \mathbb{E}_{p(\epsilon)}[f(\mathbf{z})] \tag{2.22} \label{eq:2_22} \\
&= \mathbb{E}_{p(\epsilon)}[f(\nabla_\phi \mathbf{z})] \tag{2.23} \label{eq:2_23} \\
&\simeq \nabla_\phi f(\mathbf{z}) \tag{2.24} \label{eq:2_24}
\end{align}
$$

where $$\mathbf{z} = g(\epsilon, \phi, \mathbf{x})$$ with random noise sample $$\epsilion \; \sim \; p(\epsilon)$$

#### Gradient of ELBO

위에서 말한 reparameterization trick을 사용해서 ELBO를 다시 써보면

$$
\begin{align}
\mathcal{L}_{\theta, \phi}(\mathbf{x}) &= \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} [\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})] \tag{2.25} \label{eq:2_25} \\
&= \mathbb{E}_{p(\epsilon)} [\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})] \tag{2.26} \label{eq:2_26}
\end{align}
$$

where $$\mathbf{z} = g(\epsilon, \phi, \mathbf{x})$$

Monte Carlo 추정량 $$\tilde{\mathcal{L}}_{\theta, \phi}(\mathbf{x})$$도 다시 쓸 수 있다.

$$
\begin{align}
\epsilon \; &\sim \; p(\epsilon) \tag{2.27} \label{eq:2_27} \\
\mathbf{z} &= g(\epsilon, \phi, \mathbf{x}) \tag{2.28} \label{eq:2_28} \\
\tilde{\mathcal{L}}_{\theta, \phi}(\mathbf{x}) &= \log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}) \tag{2.29} \label{eq:2_29}
\end{align}
$$

##### Unbiasedness

ELBO의 gradient 추정량은 불편추정량이고 noise $$\epsilon \; &\sim \; p(\epsilon)$$가 평균으로 분포돼있다. single datapoint ELBO gradient와 똑같이 나온다.

$$
\begin{align}
\mathbb{E}_{p(\epsilon)}[ \nabla_{\theta, \phi} \tilde{\mathcal{L}}_{\theta, \phi}(\mathbf{x};\epsilon)] &= \mathbb{E}_{p(\epsilon)} [\nabla_{\theta, \phi} (\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}))] \tag{2.30} \label{eq:2_30} \\
&= \nabla_{\theta, \phi} \mathbb{E}_{p(\epsilon)} [\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x})] \tag{2.31} \label{eq:2_31} \\
&= \nabla_{\theta, \phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})
\end{align} \tag{2.32} \label{eq:2_32}
$$

#### Computation of $$\log q_\phi(\mathbf{z} \mid \mathbf{z})$$

ELBO의 추정량 계산은 $$\log q_\phi(\mathbf{z} \mid \mathbf{z})$$의 density 계산이 필요하다.(given a value of $$\mathbf{x}$$ and given a value of $$\mathbf{z}$$ 또는 마찬가지로 $$\epsilon$$)

이 log-density 계산은 $$g()$$의 선택에 따라 간단해 질 수 있다.

일반적으로 $$p(\epsilon)$$의 density는 알고 있다.

$$g(.)$$가 invertible 함수이면 $$\epsilon$$과 $$\mathbf{z}$$의 density들은 다음과 같이 관련되어 있다.

$$
\log q_\phi(\mathbf{z} \mid \mathbf{z}) = \log p(\epsilon) - \log d_\phi(\mathbf{x}, \epsilon) \tag{2.33} \label{eq:2_33}
$$

$$\log d_\phi(\mathbf{x}, \epsilon)$$은 Jacobian matrix$$(\partial \mathbf{z}  / \partial \epsilon)$$의 determinant의 절댓값이다.

$$
\log d_\phi(\mathbf{x}, \epsilon) = \log \left\vert \det(\frac{\partial \mathbf{z}}{\partial \epsilon}) \right\vert \tag{2.34} \label{eq:2_34}
$$

이것을 $$\epsilon$$에서 $$\mathbf{z}$$ 변화의 log-determinant라고 부른다.

$$
\frac{\partial \mathbf{z}}{\partial \epsilon} = \frac{\partial (z_1, \cdots, z_k)}{\partial (\epsilon_1, \cdots, \epsilon_k)} = 
  \begin{pmatrix}
  \frac{\partial z_1}{\partial \epsilon_1} & \cdots & \frac{\partial z_1}{\partial \epsilon_k} \\
  \vdots  & \ddots & \vdots  \\
  \frac{\partial z_k}{\partial \epsilon_1} & \cdots & \frac{\partial z_k}{\partial \epsilon_k} \tag{2.35} \label{eq:2_35}
  \end{pmatrix}
$$

나중에 이 $$g()$$를 $$\log d_\phi(\mathbf{x}, \epsilon)$$의 계산이 쉬우면서 flexible한 inference 모델 $$q_\phi(\mathbf{z} \mid \mathbf{x})$$를 갖도록 설정해 줄 것이다.

<br>

### Factorized Gaussian Posteriors

일반적으로 간단한 factorized Gaussian encoder를 선택한다.

$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} ; \mu, \text{diag}(\sigma^2))$$ :

$$
\begin{align}
(\mu, \log \sigma) &= EncoderNeuralNet_\phi(\mathbf{x}) \tag{2.36} \label{eq:2_36} \\
q_\phi(\mathbf{z} \mid \mathbf{x}) &= \prod_i q_\phi(\mathbf{z}_i \mid \mathbf{x}) = \prod_i\mathcal{N}(\mathbf{z}_i ; \mu_i ; \sigma_i^2) \tag{2.37} \label{eq:2_37}
\end{align}
$$

where $$\mathcal{N}(\mathbf{z}_i ; \mu_i ; \sigma_i^2) $$ is p.d.f of univariate Gaussian distribution.

reparameterization 후에 다시 쓸 수 있다.

$$
\begin{align}
\epsilon \; &\sim \; \mathcal{N}(0, \mathbf{I}) \tag{2.38} \label{eq:2_38} \\
(\mu, \log \sigma) &= EncoderNeuralNet_\phi(\mathbf{x}) \tag{2.39} \label{eq:2_39} \\
\mathbf{z} &= \mu + \sigma \odot \epsilon \tag{2.40} \label{eq:2_40}
\end{align}
$$

where $$\odot$$ is element-wise product

Jacobian은 
$$
\frac{\partial \mathbf{z}}{\partial \epsilon} = \text{diag}(\sigma) \tag{2.41} \label{eq:2_41}
$$

이다.

diag는 diagonal matrix이다. diagonal matrix의 determinant는 diagonal들의 곱이므로 Jacobian의 log determinant는 다음과 같이 계산된다.

$$
\log d_\phi(\mathbf{x}, \epsilon) = \log \left\vert \det(\frac{\partial \mathbf{z}}{\partial \epsilon}) \right\vert = \sum_i \log \sigma_i \tag{2.42} \label{eq:2_42}
$$

그리고 posterior density는

$$
\begin{align}
\log q_\phi(\mathbf{z} \mid \mathbf{x}) &= \log p(\epsilon) - \log d_\phi(\mathbf{x}, \epsilon) \tag{2.43} \label{eq:2_43} \\
&= \sum_i \log \mathcal{N}(\epsilon_i ; 0,1) - \log \sigma_i \tag{2.44} \label{eq:2_44}
\end{align}
$$

이다. when $$\mathbf{z} = g(\epsilon, \phi, \mathbf{x})$$











---
title: Inverse Autoregressive Flow
updated: 2021-01-25 17:49
---

([Variational Inference](https://hanbumko.github.io/Variational-Inference) 챕터5입니다.)

## 5. Inverse Autoregressive Flow

inference model의 flexibility를 향상시키면 ELBO의 tightness도 향상된다. 이번 챕터에서는 inference model $$q_\phi(\mathbf{z} \mid \mathbf{x})$$의 flexibility를 향상시키는 방법들을 알아본다.

<div class="divider"></div>

### 5.1. Requirements for Computational Tractability

ELBO를 효과적으로 최적화하기 위한 요구조건은 두 가지가 있다.

1. $$q_\phi(\mathbf{z} \mid \mathbf{x})$$ 미분의 계산 효율성
2. 샘플링의 계산 효율성

계산이 오래걸려 Gaussian posterior를 사용 경우가 있는데, 좀 더 flexible한 posterior가 필요하다.

<div class="divider"></div>

### 5.2. Improving the Flexibility of Inference Models

우선 flexibility를 높이는 일반적인 테크닉인 auxiliary latent variable과 normalizing flows를 다룰 것이다.

#### 5.2.1. Auxiliary Latent Variables

continuous한 보조 변수 $$\mathbf{u}$$를 inference model과 generative model에 추가시키는 방법이다.

inference model은 $$\mathbf{u}$$와 $$\mathbf{z}$$에 대한 분포를 정의하고 다음과 같이 분해될 수 있다.

$$
\begin{align}
q_\phi(\mathbf{u}, \mathbf{z} \mid \mathbf{x}) = q_\phi(\mathbf{u} \mid \mathbf{x})q_\phi(\mathbf{z} \mid \mathbf{u}, \mathbf{x}) \tag{5.1} \label{eq:5_1}
\end{align}
$$

$$
\begin{align}
q_\phi(\mathbf{z} \mid \mathbf{x}) = \int q_\phi(\mathbf{u}, \mathbf{z} \mid \mathbf{x})d\mathbf{u} \tag{5.2} \label{eq:5_2}
\end{align}
$$

마찬가지로 generative model도 $$\mathbf{x}, \mathbf{z}, \mathbf{u}$$의 joint distribution을 다음과 같이 정의할 수 있다.

$$
\begin{align}
p_\theta(\mathbf{x}, \mathbf{z}, \mathbf{u}) = p_\theta(\mathbf{u} \mid \mathbf{x}, \mathbf{z})p_\theta(\mathbf{x}, \mathbf{z}) \tag{5.3} \label{eq:5_3}
\end{align}
$$

ELBO objective도 경험분포(데이터)가 주어졌을 때 KL divergence의 최소화와 같다.

$$
\begin{align}
\mathbb{E}_{q_\mathcal{D}(\mathbf{x})}  \lbrack \mathbb{E}_{q_\phi(\mathbf{u}, \mathbf{z} \mid \mathbf{x})} \lbrack \log p_\theta(\mathbf{x}, \mathbf{z}, \mathbf{u})-\log q_\phi (\mathbf{u}, \mathbf{z} \mid \mathbf{x}) \rbrack \rbrack \tag{5.4} \label{eq:5_4}\\
= D_{KL}(q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}, \mathbf{u}  \| p_\theta(\mathbf{x}, \mathbf{z}, \mathbf{u})) \tag{5.5} \label{eq:5_5}
\end{align}
$$

auxiliary variables 없을때의 경우를 다시 보면,

maximization of the original ELBO = minimization of KL divergence  $$\eqref{eq:5_5}$$
miximization of marginal likelihood = minimization of KL divergence  $$\eqref{eq:5_5}$$

다음 식들을 통해 추가적인 이해를 얻을 수 있다.

$$
\begin{align}
&D_{KL}(q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z}, \mathbf{u} \| p_\theta(\mathbf{x}, \mathbf{z}, \mathbf{u})) \tag{5.6} \label{eq:5_6} \\

&\text{(=ELBO objective with auxiliary variables)} \\

&= D_{KL}(q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z}) \| p_\theta(\mathbf{x}, \mathbf{z})) + \mathbb{E}_{q_\mathcal{D}(\mathbf{x}, \mathbf{z})} \lbrack D_{KL}(q_{\mathcal{D},\phi}(\mathbf{u} \mid \mathbf{x}, \mathbf{z}) \| p_\theta(\mathbf{u} \mid\mathbf{x}, \mathbf{z})) \rbrack \\

&\geq D_{KL}(q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z}) \| p_\theta(\mathbf{x}, \mathbf{z})) \tag{5.7} \label{eq:5_7} \\

&\text{(=original ELBO objective)} \\

&= D_{KL}(q_\mathcal{D}(\mathbf{x}) \| p_\theta(\mathbf{x})) + \mathbb{E}_{q_\mathcal{D}(\mathbf{x})} \lbrack D_{KL}(q_{\mathcal{D},\phi}(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x})) \rbrack \tag{5.8} \label{eq:5_8} \\

&\geq D_{KL}(q_\mathcal{D}(\mathbf{x}) \| p_\theta(\mathbf{x})) \tag{5.9} \label{eq:5_9} \\

&\text{(=Marginal log-likelihood objective)}
\end{align}
$$

식을보면 auxiliary variable $$\mathbf{u}$$를 사용할 때 ELBO가 더 안좋은 것 같이 보인다.

$$
D_{KL}(q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z}, \mathbf{u} \| p_\theta(\mathbf{x}, \mathbf{z}, \mathbf{u})) \geq D_{KL}(q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z} \| p_\theta(\mathbf{x}, \mathbf{z}))
$$

하지만 flexibility를 높이는 것이 KL divergence의 추가적인 cost보다 더 중요하다.(비중이 크다.)

#### Normalizing Flows

또다른 방법인 Normalizing Flow (NF)의 일반적인 아이디어는 random한 변수로 시작하여 알려져있고 계산이 쉬운 분포와 같이 시작하고, invertible parameterized transformations $$f_t$$를 연쇄적으로 적용하는 식이다. 그러면 마지막 iterate의 $$\mathbf{z}_T$$는 좀 더 flexible한 분포를 갖는다.

$$
\begin{align}
&\epsilon_0  \sim  p(\epsilon) \tag{5.10} \label{eq:5_10} \\
&\text{for t = 1 ... T:} \tag{5.11} \label{eq:5_11} \\
&\; \;  \epsilon_t = f_t(\epsilon_{t-1}, \mathbf{x}) \tag{5.12} \label{eq:5_12} \\
&\mathbf{z} = \epsilon_t \tag{5.13} \label{eq:5_13}
\end{align}
$$

이 transformation의 Jacobian은 다음과 같이 인수분해 된다.

$$
\frac{d\mathbf{z}}{d\epsilon_0} =  \prod_{t=1}^T \frac{d\epsilon_t}{d\epsilon_{t-1}} \tag{5.14} \label{eq:5_14}
$$

따라서 determinant 또한 인수분해 된다.

$$
\log \left\lvert \det \big( \frac{d\mathbf{z}}{d\epsilon_0} \big) \right\rvert =  \sum_{t=1}^T \log \left\lvert \det \big( \frac{d\epsilon_t}{d\epsilon_{t-1}} \big) \right\rvert \tag{5.15} \label{eq:5_15}
$$

transformation $$f_t$$의 Jacobian의 determinant가 계산될 수 있으면 $$z$$의 p.d.f.도 구할 수 있다.

$$
\log q_\phi (\mathbf{z} \vert \mathbf{x}) = \log p(\epsilon_0) - \sum_{t=1}^T \log \left\lvert \det \big( \frac{d\epsilon_t}{d\epsilon_{t-1}} \big) \right\rvert \tag{5.16} \label{eq:5_16}
$$

<div class="divider"></div>

### 5.3. Inverse Autoregressive Transformations

$$\mathbf{y}$$를 MADE나 PixelCNN처럼 고차원 공간을 다루는 normalizing flows에 의해 나온 변수라 하자. $$\mathbf{y} = \lbrace \mathcal{y}_i \rbrace_{i=1}^D$$ (원소에 정해진 순서가 존재)

$$\lbrack \mu(\mathbf{y}), \sigma(\mathbf{y}) \rbrack$$는 $$\mathbf{y}$$에서 $$\mu, \sigma$$벡터로 가는 함수를 나타내기로 한다.

autoregressive 구조에 의해 위 함수의 Jacobian은 diagonal이 0인 triangular 행렬이다.
$$\partial \lbrack \mu_i, \sigma_i \rbrack  / \partial \mathbf{y}_j = [0, 0] \; \text{for} \; j \geq i$$

$$\lbrack \mu(\mathbf{y}_{1:i-1}), \sigma(\mathbf{y}_{1:i-1}) \rbrack$$은 $$\mathbf{y}$$에서 이전 원소들의 함수의 mean과 std의 추정치이다.

위와 같은 모델은 noise vector $$\epsilon \; \sim \; \mathcal{N}(0, \mathbf{I})$$ 에서 벡터 $$\mathbf{y}$$로의 sequential transformation이다.

$$
\begin{align}
\mathbf{y} : \mathcal{y}_0 &= \mu_0 + \sigma_0 \odot \epsilon_0 \\
\mathcal{y}_i &= \mu_i(\mathbf{y}_{1:i-1}) + \sigma_i(\mathbf{y}_{1:i-1}) \cdot \epsilon_i
\end{align}
$$

모든 $$i$$에 대해 $$\sigma_i > 0$$ 이면 sampling은 일대일 transformation이고, inverted 될 수 있다.

$$
\epsilon_i = \frac{\mathcal{y}_i - \mu_i(\mathbf{y}_{1:i-1})}{\sigma_i(\mathbf{y}_{1:i-1})} \tag{5.18} \label{eq:5_18}
$$

이 역변환(inverse transformation)은 parallelized 될 수 있고 (autoregressive autoencoder의 경우), $$\epsilon$$의 각 원소 $$\epsilon_i$$가 서로 독립적이다. (do not depend on each other.) 벡터로 표현하면 다음과 같이 표현 가능하고,

$$
\epsilon = (\mathbf{y}-\mu(\mathbf{y})) / \sigma(\mathbf{y}) \tag{5.19} \label{eq:5_19}
$$

빼기와 나누기는 element-wise 이다.

이 inverse autoregressive 연산은 간단한 Jacobian determinant를 갖는다. 
$$\partial \lbrack \mu_i, \sigma_i \rbrack  / \partial \mathbf{y}_j = [0, 0] \; \text{for} \; j \geq i$$

결과적으로 변환은, lower triangular Jacobian $$(\partial \epsilon_i / \partial \mathcal{y}_j = 0 \; \text{for} \; j>i)$$이 간단한 diagonal을 갖는다.
$$(\partial \epsilon_i / \partial \mathcal{y}_i = \frac{1}{\sigma_i})$$

따라서 log-determinant는

$$
\log \det \lvert \frac{d\epsilon}{d\mathbf{y}} \rvert = \sum_{i=1}^D - \log \sigma_i(\mathbf{y}) \tag{5.20} \label{eq:5_20}
$$

이다.

model flexibility의 조합, dimension에 관한 parallelizability, 간단한 log-determinant가 고차원에 대한 normalizing flow를 다룰수 있게 한다.

이 이후로는 조금 다르지만 똑같은 변환을 사용한다.

$$
\begin{align}
\epsilon = \sigma(\mathbf{y})\odot\mathbf{y} + \mu(\mathbf{y}) \tag{5.21} \label{eq:5_21} \\
\log \det \lvert \frac{d\epsilon}{d\mathbf{y}} \rvert = \sum_{i=1}^D  \log \sigma_i(\mathbf{y}) \tag{5.22} \label{eq:5_22}
\end{align}
$$

<div class="divider"></div>

### 5.4. Inverse Autoregressive Flow (IAF)

첫번째 encoder NN의 output $$\mu_0, \sigma_0$$(더하여 추가적인 output $$\mathbf{h}$$도)는 뒤의 모델들에 순차적으로 사용된다. 변환들은 factorized Gaussian $$q_\phi(\mathbf{z}_0 \vert \mathbf{x}) = \mathcal{N}(0, \text{diag}(\sigma^2))$$으로 초기화 된다.

$$
\begin{align}
\epsilon_0 \; &\sim \; \mathcal{N}(0, \mathbf{I}) \tag{5.23} \label{eq:5_23} \\
(\mu_0, \log \sigma_0, \mathbf{h}) &= \text{EncoderNeuralNet}(\mathbf{x};\theta) \tag{5.24} \label{eq:5_24} \\
\mathbf{z}_0 &= \mu_0 + \sigma_0 \odot \epsilon_0 \tag{5.25} \label{eq:5_25}
\end{align}
$$

IAF는 다음의 변환들의 연속이다.

$$
\begin{align}
(\mu_t, \sigma_t) &= \text{AutoregressiveNeuralNet}_t(\epsilon_{t-1}, \mathbf{h} ; \theta) \tag{5.26} \label{eq:5_26} \\
\epsilon_t &= \mu_t + \sigma_t \odot \epsilon_{t-1} \tag{5.27} \label{eq:5_27}
\end{align}
$$

![algorithm5](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/algorithm5.png?raw=true)

마지막 iterate에서 식$$\eqref{eq:5_16}$$을 이용해 density를 구할 수 있다.

$$
\begin{align}
\mathbf{z} &\equiv \epsilon_T \tag{5.28} \label{eq:5_28}\\
\log q(\mathbf{z} \vert \mathbf{x}) &= -\sum_{i=1}^D (\frac{1}{2}\epsilon_i^2 + \frac{1}{2}\log(2\pi) + \sum_{t=0}^T \log \sigma_{t,i}) \tag{5.29} \label{eq:5_29}
\end{align}
$$

![figure5_1](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/figure5_1.png?raw=true)

LSTM처럼 두 개의 unconstrained 실수 벡터 $$(\mathbf{m}_t, \mathbf{s}_t)$$를 이용할 수도 있다.

$$
\begin{align}
(\mathbf{m}_t, \mathbf{s}_t) &= \text{AutoregressiveNeuralNet}_t(\epsilon_{t-1}, \mathbf{h};\theta) \tag{5.30} \label{eq:5_30} \\

\sigma_t &= \text{sigmoid}(\mathbf{s}_t) \tag{5.31} \label{eq:5_31} \\

\epsilon_t &= \sigma_t \odot \epsilon_{t-1} + (1-\sigma_t) \odot \mathbf{m}_t \tag{5.32} \label{eq:5_32}
\end{align}
$$

-> Algorithm 5에 나와 있다.

![figure5_2](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/variational_inference/figure5_2.png?raw=true)

$$\mathbf{s}_t$$가 충분히 positive한 값 (+1 혹은 +2)가 나오도록 초기화 하는것이 좋다고 알려져 있다.

<div class="divider"></div>



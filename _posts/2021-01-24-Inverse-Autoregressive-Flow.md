---
title: Inverse Autoregressive Flow
updated: 2021-01-24 17:42
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







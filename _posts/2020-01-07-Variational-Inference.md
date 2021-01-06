---
title: Variational Inference
updated: 2021-01-07 00:33
---



# Variational Inference

랩 내의 스터디중 하나에서 [Kingma](http://www.dpkingma.com/)의 Thesis를 공부하고 있는데, 정리도 할겸 복습용으로 작성중인 글입니다. [Full Thesis](https://pure.uva.nl/ws/files/17891313/Thesis.pdf)




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
\mathbf{x} \quad \sim \quad p_\theta (\mathbf{x})
$$

<br>

이 true distribution $$p^\ast (\mathbf{x})$$의 파라미터를 추론하는 과정을 **Learning**이라 한다.

$$
p_\theta (\mathbf{x}) \approx p^\ast (\mathbf{x})
$$

$$p_\theta (\mathbf{x})$$는 충분히 flexible 하고 충분히 정확하게 모델링 될 수 있어야 한다.












---
title: Combinatorial Optimization
updated: 2021-01-04 21:11



---



## Paper list

* [Learning Combinatorial Optimization Algorithms over Graphs](#learning-combinatorial-optimization-algorithms-over-graphs) [^1]


<div class="divider"></div>


# Learning Combinatorial Optimization Algorithms over Graphs

* 그래프 임베딩과 RL 이용한 문제 해결
* 그래프 임베딩으로 Action 정해줌, MDP 세팅
* Minimum Vertex Cover, Maximum Cut, TSP 문제에 적용함 (일반적인 문제에 적용가능)



## Traditional Approaches

* Exact algorithm
  - 열거, branch-and-bound(integer programming formulation) -> 문제가 크면 못 풀 정도로 시간 소요

* Approximation
  - Polynomial-time approximation algorithms -> 이상적이나 Optimality 보장되지 않음, 존재하지 않을 수 있음
* Heuristics
  - 효과적이고 빠르나 이론적인 근거 존재하지 않음, 알고리즘 만들때 많은 trial and error, problem-specific함



그래프 최적화문제 $${G}$$가 주어지고 데이터 분포 $${D}$$를 모를때 heuristics 보다 나은 알고리즘으로 학습할 수 있을까?



이전 페이퍼들과 다른 점

* Algorithm design pattern
  - greedy로 디자인 하였음. Graph constraints 존재
* Algorithm representation
  - 그래프 임베딩으로 **structure2vec**을 사용
* Algorithm training
  - n-step Q-learning, Fitted Q-iteration 사용 (delayed reward 고려)



weighted graphs $${ G(V,E,w) }$$ 로 표현가능

$${V}$$ -> Vertex, Node

$${E}$$ -> Edge

$${w}$$ -> weight

함수 $${w: E\rightarrow\mathbb{R^+}}$$

$$w(u,v)$$ , $$\left( u,v \right) \in E$$



##### Minimum Vertex Cover (MVC)

* **최소한의 노드**로 모든 edge 찾는 문제

##### Maximum Cut (MAXCUT)

* Cut-set의 합이 최대화 되도록 **edge의 개수**를 찾는 문제

##### Traveling Salesman Problem (TSP)

* 모든 node를 한번씩만 돌면서 weight를 최소화 하는 문제. 여행경로.



문제들에 대해 알고리즘을 일반적으로 표현 가능

1. A problem instance $${ G }$$  ~  $${ D }$$

2. Partial Solution -> ordered list

   $$S=\left( v_{ 1 },\quad { v }_{ 2 },\quad \cdots ,\quad { v }{ \left| S \right|  } \right)$$

   $$\overline { S }$$ : 추가가능한 노드들, given S

   binary decision variables $${x}$$, $${x}_{v}=1$$ if $$v\in S$$, $$x_v=0$$ otherwise

3. A maintenance procedure $$h(S)$$ -> maps $${S}$$ to constraints

4. quailityt of partial solution S objective function $$C(h(s), G)$$

5. Greedy 알고리즘을 이용해 node $$v$$ 선택 -> $$Q(h(S), v) \in \mathbb{R}$$를 최대화 시키는 노드
   그리고 선택된 $$V^*$$를 $$S$$에 추가

   $$S:=\left( S, { v }^{ * } \right) ,\quad where\quad { v }^{ * }:=\underset { v\in \bar S }{ argmax }Q\left( h(S), v \right)$$

6. termination criterion $$t\left( h(S) \right)$$ 가 만족될때까지 반복

모든 문제가 다른 $$h(S)$$, $$c$$, $$t$$ 들로 표현 가능



##### MVC 

helper function x,

$$
C\left( h\left(S\right),G \right) =-\left| S \right|
$$

$$t$$: 모든 엣지 사용 확인



##### MAXCUT

helper function divides $$V$$ into two sets $$\left( S, \overline { S }  \right)$$ & maintains a cut-set
$$C=\left\{ \left( u,v \right) |\left( u,v \right) \in E,\quad u\in S,\quad v\in \overline { S }  \right\} $$,

$$c\left( h(S),G \right) =\sum _{ (u,v)\in C }^{  }{ w\left( u,v \right)  } $$,

termination x



##### TSP

$$h$$ maintains tour (node orders),

$$
C\left( h(S), G \right) =-\sum_{ i=1 }^{ \left| S \right| -1 }{ w\left( S(i),S(i+1) \right) -w\left( S(\left| S \right| ),S(1) \right)  }
$$

terminate when $$S=V$$




##### Representation: Graph Embedding

**structure2vec** : 각 노드들은 p-dimension의 feature embedding $$\mu_v$$ 갖고있음, $$v\in V$$

$$
\mu_v^{t+1}\quad\leftarrow \quad F(x_v, \; \left\{\mu_u^{(t)} \right\}_{u\in N(v)}, \;  \left\{ w(v,u)\right\}_{u\in N(v)}  ; \theta )
$$

$$F$$: generic non-linear mapping(NN, kernel function)

$$N(v)$$: set of neighbors

-> 이웃 노드들의 feature, weight를 이용해 새로운 $$\mu_v^{t+1}$$ 생성

-> 위 연산을 $$T$$번 반복하면 T-hop의 정보를 얻는 것으로 생각할 수 있음 (usually $$T=4$$)

뉴럴넷 이용해 다시 적어보면, 위 식은

$$
relu(\theta _{ 1 }x_{ v }\, +\, \theta _{ 2 }\sum _{ u\in N(v) } \mu_u^{(t)} \, + \, \theta_3\sum_{u\in N(v)}{relu(\theta_4w(v,u))}  )
$$

로 표현할 수 있다.

$$\theta_1\in\mathbb{R}^\rho \quad$$,
$$\theta_2,\theta_3\in\mathbb{R}^{\rho \times \rho \quad }$$, 
$$\theta_4\in\mathbb{R}^\rho$$

각 파라미터는 위와 같은 차원을 갖고 차근차근(?) 따져보면 $$\mu_v^{t+1}$$는 $$\rho$$ 차원을 갖는것을 확인할 수 있다.

위 연산은 각 노드들의 feature를 구하는 과정이고 이 구해진 feature들을 이용해 estimated Q 또한 뉴럴넷 이용해 구한다.

$$
\hat{Q}(h(S), v;\theta) \, = \, \theta_5^Trelu([\theta_6\sum_{u\in V}{\mu_u^{(T)}},\; \theta_7\mu_v^{(T)}])
$$

$$\theta_5\in \mathbb{R}^{2\rho} \quad$$,
$$\theta _{ 6 },\theta _{ 7 }\in { R }^{ \rho \times \rho } \quad$$,
$$[\cdot , \cdot ]: Concatenation$$

-> 해당 노드의 feature + 모든 feature의 합(global feature)를 concatenation 한다.




##### Training: Q-learning

앞에서 정의했던 $$\hat{Q}$$를 RL의 state-value function으로 간주

$$\hat{Q}$$는 다른 분포, 사이즈를 갖고있는 $${ D },\; D=\left\{ G_i \right\}_{i=1}^m$$에 대해 학습함




##### RL formulation

- States: $$\sum_{v\in V}{\mu_v}$$

- Transition: deterministic (MDP세팅)

- Actions: a node of G not in S

- Rewards:

$$
r(s,v) = c(h(S^\prime), G) - c(h(S), G) \\
c(h(\phi), G)=0
$$

  -> 새로운 state에서의 cost는 커져야 하고 현재의 state에서의 cost는 작아야 함
  -> 현재 cost가 작게 되도록 선택해야 함

- Policy: greedy policy

$$
\pi(v \mid s) := \underset { j\in \bar S }{ argmax }\hat{Q}(h(S), v^\prime)
$$

optimal Q-function은 $$Q^\ast$$로 표기




| Problem | State                           | Action                | Helper function     | Reward               | Termination                   |
| ------- | ------------------------------- | --------------------- | ------------------- | -------------------- | ----------------------------- |
| MVC     | subset of nodes selected so far | add node to subset    | None                | -1                   | all edges are covered         |
| MAXCUT  | subset of nodes selected so far | add node to subset    | None                | change in cut weight | cut weight cannot be improved |
| TSP     | partial tour                    | grow tour by one node | Insertion operation | change in tour cost  | tour includes all nodes       |




##### Learning algorithm

n-step Q-learning, fitted Q-iteration

노드를 다 추가하면 한 에피소드가 끝난것으로 간주,

하나의 노드를 선택하는 것을 1-step 이라고 표현



1-step Q-learning은 매 스텝마다 업데이트
n-step Q-learning은 n 스텝마다 업데이트 (for delayed reward)
$$
y=\sum_{i=0}^{n-1}{r(s_{t+i}, \, v_{t+i}) + \gamma max_{v^{\prime}} \hat{Q}(h(S_{t+n}), v^{\prime};\theta ) }
$$



fitted Q-iteration은 버퍼에 저장했다가 배치로 꺼내서 업데이트 하는것



##### Algorithm

![algorithm_1](https://github.com/HanbumKo/HanbumKo.github.io/blob/master/_posts_imgs/combinatorial_optimization/algorithm1.png)




<div class="divider"></div>







[^1]: [https://arxiv.org/abs/1704.01665]

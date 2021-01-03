---
title: Combinatorial Optimization
updated: 2021-01-04 21:11

---



## Paper list

* Learning Combinatorial Optimization Algorithms over Graphs [^1]



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
  - Fitted Q-learning 사용 (delayed reward 고려)



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

   $$S:=\left( S, { v }^{ * } \right) ,\quad where\quad { v }^{ * }:={ argmax }_{ v\in \overline { s }  }Q\left( h(S), v \right)$$

6. termination criterion $$t\left( h(S) \right)$$ 가 만족될때까지 반복

모든 문제가 다른 $$h(S)$$, $$c$$, $$t$$ 들로 표현 가능



##### MVC 

helper function x, 

$$C\left( h(S), G \right) = -\left| S \right|$$,

$$t$$: 모든 엣지 사용 확인



##### MAXCUT

helper function divides $$V$$ into two sets $$\left( S, \overline { S }  \right)$$ & maintains a cut-set $$C=\left\{ \left( u,v \right) |\left( u,v \right) \in E,\quad u\in S,\quad v\in \overline { S }  \right\} $$,

$$c\left( h(S),G \right) =\sum _{ (u,v)\in C }^{  }{ w\left( u,v \right)  } $$,

termination x



##### TSP

$$h$$ maintains tour (node orders),

$$C\left( h(S),\quad G \right) =-\sum _{ i=1 }^{ \left| S \right| -1 }{ w\left( S(i),S(i+1) \right) -w\left( S(\left| S \right| ),S(1) \right)  } $$,

terminate when $$S=V$$

| Problem | State                           | Action                | Helper function     | Reward               | Termination                   |
| ------- | ------------------------------- | --------------------- | ------------------- | -------------------- | ----------------------------- |
| MVC     | subset of nodes selected so far | add node to subset    | None                | -1                   | all edges are covered         |
| MAXCUT  | subset of nodes selected so far | add node to subset    | None                | change in cut weight | cut weight cannot be improved |
| TSP     | partial tour                    | grow tour by one node | Insertion operation | change in tour cost  | tour includes all nodes       |



(TBA)











[^1]: [https://arxiv.org/abs/1704.01665](https://arxiv.org/abs/1704.01665)


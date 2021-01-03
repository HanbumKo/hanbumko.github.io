---
title: Combinatorial Optimization
updated: 2021-01-04 21:11
---



Paper list

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



Given a graph optimization problem $${G}$$ and a distribution $${D}$$ of problem instances, can we learn better heuristics that generalize to unseen instances from D?

















[^1]:[https://arxiv.org/abs/1704.01665]


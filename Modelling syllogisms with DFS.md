# Modelling syllogisms with DFS

I tried to sample a DFS space that can give rise to logically valid reasoning patterns in syllogisms based on the Ragni 2016 dataset. A syllogism in this dataset is a tuple of two premises and a total of three terms,  where each statement is of Aristotelean form . This means that it is quantified using one of four quantifiers: $A: $ "All" $I:$ "Some" $E:$ "No" $O:$ "Some not" Each syllogism has zero or more associated logically valid conclusions that necessarily follow from the premises. An example of such a syllogism is as follows:

> All a are b
>
> All b are c
>
> ---------------
>
> What, if anything, follows?

The syllogism above contains two $A$ type quantifiers , AA for short. The ordering of the terms $\{a, b, c\}$ matters for the set of valid conclusions. In total 4 meaningfully different term-orderings are possible: AB, BC (1), AB, CA (2),  AB, CB (3) and AB, AC (4). According to this coding scheme, the example above is coded as AA1. This coding scheme gives rise to 64 unique syllogism templates. Conclusions are similarly structured. They are single quantified statements consisting of the terms a and c. An example valid conclusion to AA1 is Aac (All a are c). Some syllogisms do not have any valid conclusions, and hence the target response to these cases is "No Valid Conclusion". This results in a set of 9 unique conclusions.

My initial aim is to replicate a purely logical system in DFS. That is, a system that gives rise to only the logically valid conclusions for each syllogism. In order to translate syllogisms into DFS, we require a DFS space that assigns truth-values to the three unique terms of each syllogism across different models. Then deriving the semantics of the premises and conclusions is intuitive as they can be compositionally derived using the formalization of first-order-logic provided in DFS. The basic ingredients of the meaning space are 3 atomic predicates $P$ and a fixed number $n$ of entities $E$. The meaning space used in the present work is instantiated with 6 entities $\{$chet, miles, sarah, ella, nat, bill$\}$ and the three predicates $\{$Pianist(x), Singer(x), Trumpeter(x)$\}$. The following constraints are imposed on the sampling process of the meaning space.

> Constraint 1: $\forall e. \text{Pianist(e)} \vee \text{Singer(e)} \vee \text{Trumpeter(e)}$ 
>
> Constraint 2: $\exists x \exists y \exists z. \text{Pianist}(x) \wedge \text{Singer}(y) \wedge \text{Trumpeter}(z) $

In order to capture the interactions between quantifiers, for each entity $e$ there must be at least one predicate $A \in P$ such that $A(e)$ is true in all models. In words: everyone must be *something*. This is what constraint 1 guarantees. Secondly, given that we first want to model a strictly logical system, we must make sure to replicate the ontological commitment of first-order-logic that quantification is only performed over all non-empty domains. To illustrate the need for this commitment, consider the sentences $\forall e. Pianist(e) \rightarrow Singer(e)$ "All pianists are singers" and $\exists e. Pianist(e) \wedge Singer(e)$ "Some pianists are singers". Intuitively the former statement entails the latter. However, in the case that there are no pianists at all in our domain, the universal statement still holds true whilst the existential statement is false. Thus we must make sure that in each model we sample there are no empty domains. There are no explicit probabilistic constraints on the world, meaning that an entity is sampled to be any type of instrumentalist with a uniform probability.

I sample a world of 1000 models according to the constraints and constants sketched above. The semantics of the premises and conclusions are derived using the existing implementation of first order logic in vanilla DFS. Each of the possible statements are formalized as follows:

> All A are B (A): $\forall x. A(x) \rightarrow B(x)$
>
> Some A are B (I): $\exists x. A(x) \and B(x)$
>
> No A are B (E): $\forall x. A(x) \rightarrow \neg B(x)$
>
> Some A are not B (O): $\exists x. A(x) \and \neg B(x)$

The degree to which a conclusion follows from the premises is calculated by taking the inference score between the conclusion and the conjunction of the premises. 

> $	\text{inf}(p,q) = \begin{array}{ll}
> 		\frac{P(p|q)-P(p)}{1-P(p)}  & P(p|q) > P(p) \\
> 		\frac{P(p|q)-P(p)}{P(p)} & otherwise
> 	\end{array}$

This results in a graded measure that represents logically necessary conclusions with a score of $1$, impossible conclusions with a score of $-1$, and possible conclusions with a floating point value in this range. Permuting all syllogisms, that is sets of 2 premises in all possible orders of quantifiers in the set $\{A, I, E, O\}$ with the 3 predicates $\{$Pianist(x), Singer(x), Trumpeter(x)$\}$, results in 384 different syllogisms after removing duplicates ('All A are A') and syllogisms with fewer than 3 distinct predicates (All A are B & Some B are A, etc.). For each unique syllogism template we thus have 6 unique instances (192 / 6 = 64), because each predicate A, B and C can be either instantiated as Singers, Trumpeters or Pianists. 

The figure below shows the inference score of each syllogism with each conclusion, resulting in a $9 \times 64$ matrix where each square represents the degree to which the conclusion can be inferred from the conjunction of the premises. Each square is the average inference score of the aforementioned 6 instances of that template in the meaning space. Logically valid conclusions for that specific syllogism are marked with a gray asterisk. Note that the implementation of No Valid Conclusion is presently not well-formed and the scores just represents the logically desired scores.

<img src="/home/luuk/projects/dissertation/Syllogisms/test.png" alt="offline_infs" style="zoom: 25%;" />



The figure demonstrates that, aside from the NVC case, the offline measure over the sampled meaning space is able to retrieve all the logically valid conclusions as expected. Additionally the inference score gives us a graded measure of conclusions that are only possible or unlikely, but cannot be logically ruled out. These predictions however, do not mimic the human data (still need to analyze in what way they differ). One interesting case is the Oac/Oca conlusions for the AA1 and AA2 syllogisms. These conclusions do not necessary follow according to first order logic, but are still logically likely in our meaning space. I suspect the perfect logical inferences are due to the way the space is sampled such that the set of things that are A and the set of things that are C is never the same. The next step is to formalize a proper way to represent the inference score of 'No Valid Conclusion'.


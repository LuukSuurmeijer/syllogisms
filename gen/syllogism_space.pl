:- use_module('dfs-tools/src/dfs_main.pl').

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% World Specification %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Model constants and predicates %%%

constant(Person) :-
        person(Person).

property(singer(Person)) :- person(Person).
property(trumpeter(Person)) :- person(Person).
property(pianist(Person)) :- person(Person).

%%% Constant instantiations %%%

person('chet').
person('ella').
person('miles').
person('bill').
person('sarah').
person('nat').

%%% Hard constraints %%%

%There are no empty domains
constraint(exists(x,pianist(x))).
constraint(exists(x,trumpeter(x))).
constraint(exists(x,singer(x))).

% Everyone is *something*, of course you can be multiple things
%%% Deze constraint doet niks meer in de nieuwe versie
constraint(forall(x,or(or(singer(x),trumpeter(x)),pianist(x)))).

%%% Probabilistic constraints %%%

% These constraints are to boost the prevalence of non-proper subsets
probability(pianist(P), singer(P), 0.9) :- person(P).
probability(singer(P), pianist(P), 0.9) :- person(P).

probability(_,top,0.5).

%%%%%%%%%%%%%%%%%
%%% Sentences %%%
%%%%%%%%%%%%%%%%%

% sentence consists of an NP and a VP
sentence((Sen,Sem)) :- s_simpl(Sem,Sen,[]).
s_simpl(Sem) --> np1(N,A), vp(_,N,_,A,Sem).

%%%% Constituents %%%%

%NP is always a quantified expression
%% Determiners
np1(a,A)  --> ['All'], np2(A).
np1(i,A) --> ['Some'], np2(A).
np1(e,A) --> ['No'], np2(A).
np1(o,A) --> ['Some_not'], np2(A).

%% Nouns
np2(trumpeter) --> ['trumpeters'].
np2(pianist) --> ['pianists'].
np2(singer) --> ['singers'].

%%%% Main Clause VPs %%%%

% This predicate needs to have the first NP (variable A) passed so it can do the correct semantics
% Since 'are' is constant, it does not matter for the neural network eventually. So it is realized as a null-copula
vp(are,N,B,A,Sem)    --> [] , np2(B),
        { sbj_semantics(N,S),
          build_terms(B,S,N,A,L), premiss(L,Sem,N) }.

          %%%% Semantics %%%%

% Subject Semantics
sbj_semantics(i,[chet,ella,miles,bill,sarah,nat]).
sbj_semantics(a,[chet,ella,miles,bill,sarah,nat]).
sbj_semantics(e,[chet,ella,miles,bill,sarah,nat]).
sbj_semantics(o,[chet,ella,miles,bill,sarah,nat]).


%%% Build Terms %%%
% Finds the list of terms to be quantified over, depending on the desired premis
% Independent of amount of entities
% EX: build_terms(singer, [miles, chet], 'eq', 'pianist', S) --> S = [and(singer(miles) ^ pianist(miles)), and(singer(chet) ^ pianist(chet))]

build_terms(_,[],_,_,_).
% SOME A are B
build_terms(Pred,[S|Ss],'i',A,[Sem0|T]) :- !,
        First =.. [A,S],
        Second =.. [Pred,S],
        Sem0 =.. [and,First,Second],
        build_terms(Pred,Ss,'i',A,T).
% ALL A are B
build_terms(Pred,[S|Ss],'a',A,[Sem0|T]) :- !,
        First =.. [A,S],
        Second =.. [Pred,S],
        Sem0 =.. [imp,First,Second],
        build_terms(Pred,Ss,'a',A,T).
% NO A are B
build_terms(Pred,[S|Ss],'e',A,[Sem0|T]) :- !,
        First =.. [A,S],
        Second =.. [Pred,S],
        Sem0 =.. [imp,First,neg(Second)],
        build_terms(Pred,Ss,'e',A,T).
% SOME A are NOT B
build_terms(Pred,[S|Ss],'o',A,[Sem0|T]) :- !,
        First =.. [A,S],
        Second =.. [Pred,S],
        Sem0 =.. [and,First,neg(Second)],
        build_terms(Pred,Ss,'o',A,T).

%%% Premis %%%
% Starts the process of finding the correct semantics for each type of premis
premiss([H|T], Result, 'i') :-
        qtf_semantics([H|T], Result, 'i').
premiss([H|T], Result, 'a') :-
        qtf_semantics([H|T], Result, 'a').
premiss([H|T], Result, 'e') :-
        qtf_semantics([H|T], Result, 'e').
premiss([H|T], Result, 'o') :-
        qtf_semantics([H|T], Result, 'o').

%%% QTF-Semantics
% Does the actual quantification work depending on the type of premiss desired.
qtf_semantics([], _, _).
qtf_semantics([H], Conjuncts, _) :- !,
        Conjuncts = H.

qtf_semantics([H|T], or(Conj1,Conjuncts), 'i') :-
        Conj1 = H,
        qtf_semantics(T, Conjuncts, 'i').

qtf_semantics([H|T], and(Conj1,Conjuncts), 'a') :-
        Conj1 = H,
        qtf_semantics(T, Conjuncts, 'a').

qtf_semantics([H|T], and(Conj1,Conjuncts), 'e') :-
        Conj1 = H,
        qtf_semantics(T, Conjuncts, 'e').

qtf_semantics([H|T], or(Conj1,Conjuncts), 'o') :-
        Conj1 = H,
        qtf_semantics(T, Conjuncts, 'o').


%%% Generate Data %%%
gen_data(SentOut, SpaceOut, Size, Threads) :-
        dfs_sample_models_mt(Threads, Size, Ms),
        % check whether the constraints are satisfied
        % Are there non-proper subsets?
        dfs_inference_score(exists(x, and(singer(x),neg(trumpeter(x)))), and(forall(x,imp(trumpeter(x),pianist(x))),forall(x,imp(pianist(x),singer(x)))), Ms, Inf),
        Inf < 1.0,
        % Is everybody something?
        dfs_prior_probability(forall(x, or(or(singer(x),trumpeter(x)),pianist(x))),Ms,1),
        % Check syllogism for sanity
        dfs_inference_score(forall(x, imp(pianist(x),trumpeter(x))),and(forall(x,imp(pianist(x), singer(x))), forall(x,imp(singer(x), trumpeter(x)))),Ms, Ir),
        Ir is 1.0,

        dfs_localist_word_vectors(WVs), %get localist vectors for each word in the grammar
        dfs_sentence_semantics_mappings(WVs,Ms,Mappings), %generate the mappings from word vectors to semantics for each sentence
        mesh_write_set(Mappings, SentOut),
        dfs_models_to_matrix(Ms,MM),
        dfs_write_matrix(MM, SpaceOut).

gen_data(SpaceIn, SentOut) :-
        dfs_read_matrix(SpaceIn, MM),
        dfs_matrix_to_models(MM, Ms),
        % check whether the constraints are satisfied
        %dfs_inference_score(exists(x, and(singer(x),neg(trumpeter(x)))), and(forall(x,imp(trumpeter(x),pianist(x))),forall(x,imp(pianist(x),singer(x)))), Ms, Inf),
        %Inf < 1.0,
        %dfs_prior_probability(forall(x, or(or(singer(x),trumpeter(x)),pianist(x))),Ms,1),
        %dfs_inference_score(forall(x, imp(pianist(x),trumpeter(x))),and(forall(x,imp(pianist(x), singer(x))), forall(x,imp(singer(x), trumpeter(x)))),Ms, Ir),
        %Ir is 1.0,
        dfs_localist_word_vectors(WVs), %get localist vectors for each word in the grammar
        dfs_sentence_semantics_mappings(WVs,Ms,Mappings), %generate the mappings from word vectors to semantics for each sentence
        mesh_write_set(Mappings, SentOut).

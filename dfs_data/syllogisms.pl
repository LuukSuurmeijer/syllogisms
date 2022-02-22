:- use_module('../src/dfs_main.pl').

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% World Specification %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Model constants and predicates %%%

constant(Person) :-
        person(Person).

property(singer(Person)) :- person(Person).
property(trumpettist(Person)) :- person(Person).
property(pianist(Person)) :- person(Person).



%%% Constant instantiations %%%

person('chet').
person('ella').
person('miles').
person('bill').
person('sarah').
person('nat').

%%% Hard constraints %%%

% Everyone is *something*, of course you can be multiple things
constraint(forall(x,or(or(singer(x),trumpettist(x)),pianist(x)))).

%%% Probabilistic constraints %%%

% Otherwise: coin flip
probability(_,top,0.5).

%%%%%%%%%%%%%%%%%
%%% Sentences %%%
%%%%%%%%%%%%%%%%%

sentence((Sen,Sem)) :- s_simpl(Sem,Sen,[]).
s_simpl(Sem) --> np1(N,A), vp(_,N,_,A,Sem).

%%%% Constituents %%%%

%% Determiners
np1(a,A)  --> ['all'], np2(A).
np1(i,A) --> ['some'], np2(A).
np1(e,A) --> ['no'], np2(A).
np1(o, A) --> ['some', 'not'], np2(A).

%% Nouns
np2(trumpettist) --> ['trumpettists'].
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
sbj_semantics(i,[chet,ella,miles,bill]).
sbj_semantics(a,[chet,ella,miles,bill]).
sbj_semantics(e,[chet,ella,miles,bill]).
sbj_semantics(o,[chet,ella,miles,bill]).


%%% Build Terms %%%
% Finds the list of terms to be quantified over, depending on the desired premiss
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
        Sem0 =.. [imp,First,Second],
        build_terms(Pred,Ss,'e',A,T).
% SOME A are NOT B
build_terms(Pred,[S|Ss],'o',A,[Sem0|T]) :- !,
        First =.. [A,S],
        Second =.. [Pred,S],
        Sem0 =.. [imp,First,neg(Second)],
        build_terms(Pred,Ss,'o',A,T).

%%% Premiss %%%
% Starts the process of finding the correct semantics for each type of premiss
premiss([H|T], Result, 'i') :-
        qtf_semantics([H|T], Result, 'i').
premiss([H|T], Result, 'a') :-
        qtf_semantics([H|T], Result, 'a').
premiss([H|T], neg(Result), 'e') :-
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

qtf_semantics([H|T], and(Conj1,Conjuncts), 'o') :-
        Conj1 = H,
        qtf_semantics(T, Conjuncts, 'o').


gen_data(SentOut, SpaceOut) :-
        dfs_sample_models(150, Ms),
        % One cannot read and sleep at the same time
        foreach(person(Person),
                dfs_prior_probability(or(or(singer(Person),trumpettist(Person)),pianist(Person)),Ms,1)),

        dfs_localist_word_vectors(WVs),
        dfs_sentence_semantics_mappings(WVs,Ms,Mappings),
        mesh_write_set(Mappings, SentOut),
        dfs_models_to_matrix(Ms,MM),
        dfs_pprint_matrix(MM),
        dfs_write_matrix(MM, SpaceOut).

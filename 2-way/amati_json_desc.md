
An inductive logic programming system was trained and run on the training set. The resulting theory was then applied to both the training set and the trial set. The augmented 3-way json files now contain, for each student response id, a key "amati". The structure of this additional key and value is:

'amati' +-> 'prediction' -> {'correct', 'incorrect', 'partially_correct'}
        |
        +-> 'literals' -> [ {'pred' -> 'contains',
                             'document' -> {'student_response',
                                            'question',
                                            'reference_answer'},
                             'negated' -> {'true', 'false'},
                             'lemma' -> *A lemma*,
                             'id' -> {'@1', '@2', ...}}
                             
                                ...

                             ]

where:

'prediction' is the predicted grade (corresponding to the actual grade in the original set, with case adjusted, and underscore replacing a blank in 'partially_correct'),

'literals' is a set of clauses of a rule which must be matched to award the predicted grade. The value of 'literals' is a list which contains zero or dictionaries, each of which contains the keys:

'pred' is the predicate of the literal. At the moment, 'contains' is the only available value.

'document' refers to whether the given lemma occurs (or not) in the student response, the question or the reference answer.

'negated' refers to whether the given lemma must (false) or must not (true) appear in the document.

'lemma' is the lemma itself.

'id' is the identifier of the lemma. 
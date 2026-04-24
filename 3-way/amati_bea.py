#############################
#
# For testing purposes, I've found
# that it's easier to use named arguments
# in the top level functions.
#
#


import pickle
from pyrsistent import pmap, pvector, pset
from collections import defaultdict

import pandas as pd

import subprocess
import json
import re
import itertools as iter

from operator import itemgetter


TIME_LIMIT = 60

AMATI_INDUCTION_FILE = "bea_induction.lp"
AMATI_APPLICATION_FILE = "bea_application.lp"


############################################################


def amati_text_string(*args, **kwargs):
    """Return a string of the question, answer and response
    indices, and the associated lemmas.
    """

    document_indices = f"""


question({kwargs['question_ss']['question_id']}).

reference_answer({kwargs['question_ss']['correct_id']}).
incorrect_answer({kwargs['question_ss']['incorrect_id']}).
partially_correct_answer({kwargs['question_ss']['partially_correct_id']}).

"""

    document_indices += "\n".join(
        [
            f"student_response({s.response_id}, {s.question_id})."
            for s in kwargs["responses_df"].itertuples()
        ]
    )

    lemmas_in_docs = "\n".join(
        [
            f'lemma_in_doc({t.id}, {t.i}, "{t.lemma}").'
            for t in kwargs["text_df"].itertuples()
        ]
    )

    text_output_string = f"""
% Document indices

{document_indices}

% document descriptions

{lemmas_in_docs}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

    return text_output_string


##############################


def amati_bk_string(*args, **kwargs):
    """Build a background string for the language data
    for the amati input
    """

    positive_cases = "\n".join(
        [
            f"actual_grade({t.response_id}, {t.score}, pos)."
            for t in kwargs["positive_examples_df"].itertuples()
        ]
    )

    negative_cases = "\n".join(
        [
            f"actual_grade({t.response_id}, {t.score}, neg)."
            for t in kwargs["negative_examples_df"].itertuples()
        ]
    )

    training_file_string = f"""
{amati_text_string(**kwargs)}

% positive cases

{positive_cases}

% negative cases

{negative_cases}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

    with open("clingo_program_files/bea_learning_engine.amati") as fIn:
        training_file_string += fIn.read()

    training_file_string += """    

%%% Any additional constraints."""

    if "min_precision" in kwargs:
        training_file_string += f"""

%% Individual rules must have precision of at least {kwargs['min_precision']}
:- evaluate_rule(precision, Precision), Precision<{int(kwargs['min_precision']*100)}."""

    if "min_coverage" in kwargs:
        training_file_string += f"""

%% Chosen rule must cover at least {kwargs['min_coverage']} instances.

:- total_accurate_predictions(TruePositives), TruePositives < {kwargs['min_coverage']}."""

    training_file_string += "\n\n"

    if kwargs.get("stopwords", False):
        with open("clingo_program_files/german_stopwords.amati") as fIn:
            training_file_string += fIn.read()

        training_file_string += "\n\n"

    with open(kwargs["rulesfile"]) as fIn:
        training_file_string += fIn.read()

    return training_file_string


################################3


def build_amati_training_file(*args, **kwargs):
    """Create the file that we're going to use as input for clingo"""

    # Get the amati encoding of the input data:
    amati_background = amati_bk_string(**kwargs)

    # Now, we should be able to call clingo as a subprocess using our
    # spare files rules.amati and learning_engine.amati
    #
    # Use a fairly obvious file for the moment as it's easier for
    # debugging

    with open(kwargs.get("filename", AMATI_INDUCTION_FILE), "w") as fOut:
        fOut.write(amati_background)

    return True


###########################################


def parse_clingo_training_output(str_in):
    """I assume that the output is encoded into utf-8, but is still in json format."""

    j = json.loads(str_in)

    if j["Result"] in ["OPTIMUM FOUND", "SATISFIABLE"]:
        out_ls = j["Call"][0]["Witnesses"][-1]["Value"]

        out_dict = {"result": j["Result"]}

        # Get the selected rule
        out_dict["selected_rule"] = [
            (r.group(1), r.group(0))
            for r in [re.match(r"selected_rule\((.+)\)", c) for c in out_ls]
            if r
        ][0]

        # Get the predicted grade
        out_dict["predicted_grade"] = [
            (r.group(1), r.group(0))
            for r in [re.match(r"predicted_grade\((.+)\)", c) for c in out_ls]
            if r
        ][0]

        # Get the parameters
        out_dict["parameters"] = sorted(
            [
                (int(r.group(1)), r.group(2), r.group(0))
                for r in [re.match(r"parameter\((\d+),(.+)\)", c) for c in out_ls]
                if r
            ]
        )

        # Get the positive cases covered
        out_dict["positive_covered"] = [
            r.group(1)
            for r in [re.match(r"positive_covered\((.+)\)", c) for c in out_ls]
            if r
        ]

        # Get the negative cases covered
        out_dict["negative_covered"] = [
            r.group(1)
            for r in [re.match(r"negative_covered\((.+)\)", c) for c in out_ls]
            if r
        ]

        # Get the evaluations (although we can work this out)
        out_dict["evaluations"] = {
            r.group(1): int(r.group(2))
            for r in [re.match(r"evaluate_rule\((.+),(\d+)\)", c) for c in out_ls]
            if r
        }

        return out_dict

    if j["Result"] == "UNSATISFIABLE":
        return {
            "result": "UNSATISFIABLE",
            "selected_rule": None,
            "predicted_grade": None,
            "parameters": None,
            "positive_covered": [],
            "negative_covered": [],
            "evaluation": None,
        }

    if j["Result"] == "UNKNOWN" and j["TIME LIMIT"] == 1:
        return {
            "result": "UNSATISFIABLE",
            "selected_rule": None,
            "predicted_grade": None,
            "parameters": None,
            "positive_covered": [],
            "negative_covered": [],
            "evaluation": None,
        }

    return out_ls


##########################################


def induce_once(*args, **kwargs):
    """Call clingo once on the provided positive and negative cases. Currently, there's no meaningful error checking. At some point,
    it might be worth checking that all the cases apply to the same
    question, and that all the positive and negative cases have
    an appropriate accuracy. However, that's not happening any time
    soon.
    """

    build_amati_training_file(**kwargs)

    clingo_output = subprocess.run(
        [
            "clingo",
            "--outf=2",
            f"--time-limit={kwargs.get('time_limit', TIME_LIMIT)}",
            kwargs.get("filename", AMATI_INDUCTION_FILE),
        ],
        capture_output=True,
        encoding="utf8",
    )

    out_dict = parse_clingo_training_output(clingo_output.stdout)

    # Add the DataFrames to make things easier
    out_dict["positive_df"] = (
        kwargs["positive_examples_df"]
        .merge(
            pd.DataFrame(index=out_dict["positive_covered"]).assign(covered=True),
            left_on="response_id",
            right_index=True,
            how="left",
        )
        .fillna(False, axis="columns")
    )

    # Hack to get the datatype right...
    out_dict["positive_df"]["covered"] = out_dict["positive_df"]["covered"].astype(bool)

    # Add the DataFrames to make things easier
    out_dict["negative_df"] = (
        kwargs["negative_examples_df"]
        .merge(
            pd.DataFrame(index=out_dict["negative_covered"]).assign(covered=True),
            left_on="response_id",
            right_index=True,
            how="left",
        )
        .fillna(False, axis="columns")
    )

    # Hack to get the datatype right...
    out_dict["negative_df"]["covered"] = out_dict["negative_df"]["covered"].astype(bool)

    return out_dict


#######################################

# induce_theory should keep calling induce_once until
# there's no improvemement


def induce_theory(
    stopwords=True, filename=AMATI_INDUCTION_FILE, time_limit=TIME_LIMIT, **kwargs
):

    results = []
    kw = kwargs.copy()
    uncovered_positive_examples_df = kw.pop("positive_examples_df")

    while True:

        print(len(uncovered_positive_examples_df), end=" ")

        # Call a single induction
        n = induce_once(
            positive_examples_df=uncovered_positive_examples_df,
            stopwords=stopwords,
            filename=filename,
            time_limit=time_limit,
            **kw,
        )

        # Put the results in the output structure
        results.append(n)

        # Remove the covered cases from the positive examples
        pos_df = n["positive_df"]
        uncovered_positive_examples_df = pos_df[-pos_df["covered"]].drop(
            "covered", axis="columns"
        )

        if n["result"] == "UNSATISFIABLE":
            return results

        if not n["positive_covered"]:
            return results

        assert n["result"] in ["SATISFIABLE", "OPTIMUM FOUND"]


########################################

########################################

########################################


def build_amati_application_file(rule, *args, **kwargs):
    """Create the file that we're going to use as input for clingo. The
    rule is one of the outputs from the induce_theory function, so has
    lots of bits that aren't necessarily needed for this function.

    I'll include stopwords by default.

    rule is a rule from a theory.

    kwargs should include questions_df, responses_df and text_df.

    kwargs should also include either rulesfile or rulesstring. If both
    supplied, rulesfile is used.

    """

    assert kwargs.get("rulesfile", None) or kwargs.get(
        "rulesstring", None
    ), "Require one of rulesfile or rulesstring"

    with open(kwargs.get("filename", AMATI_APPLICATION_FILE), "w") as fOut:

        fOut.write(amati_text_string(**kwargs))
        fOut.write("\n\n")

        with open("clingo_program_files/stopwords.amati") as fIn:
            fOut.write(fIn.read())
        fOut.write("\n\n")

        if kwargs.get("rulesfile", None):
            with open(kwargs["rulesfile"]) as fIn:
                fOut.write(fIn.read())
        else:
            fOut.write(kwargs["rulesstring"])
        fOut.write("\n\n")

        fOut.write(f"\n\n{rule['selected_rule'][1]}.")
        fOut.write("\n".join([f"\n\n{p[2]}." for p in rule["parameters"]]))
        fOut.write(f"\n\n{rule['predicted_grade'][1]}.")
        fOut.write(f"\n\n")

        fOut.write(
            f"""amati_apply(ResponseID):-
        selected_rule(Rule),
        covers(Rule, ResponseID).

#show amati_apply/1.
                   """
        )

    return True


########################################


def evaluate_rule(rule, *args, **kwargs):

    build_amati_application_file(rule, **kwargs)

    clingo_output = subprocess.run(
        [
            "clingo",
            "--outf=2",
            f"--time-limit={kwargs.get('time_limit', 60)}",
            kwargs.get("application_filename", AMATI_APPLICATION_FILE),
        ],
        capture_output=True,
        encoding="utf8",
    )

    j = json.loads(clingo_output.stdout)

    # Hopefully, this won't raise any errors. No doubt I'll
    # remember to do some checking when the experiments all
    # go tits up in a few days.

    assert len(j["Call"]) == 1
    assert len(j["Call"][0]["Witnesses"]) == 1

    covered_cases_ls = [
        r.group(1)
        for r in [
            re.match(r"amati_apply\((.+)\)", c)
            for c in j["Call"][0]["Witnesses"][0]["Value"]
        ]
        if r
    ]

    return covered_cases_ls


####################################################


def evaluate_theory(theory, *args, **kwargs):

    out_df = pd.DataFrame(index=kwargs["responses_df"]["response_id"])

    for i, rule in enumerate(theory):
        if rule["result"] in ["SATISFIABLE", "OPTIMUM FOUND"]:
            c = pd.Series(
                data=rule["predicted_grade"][0],
                index=evaluate_rule(rule=rule, **kwargs),
            )

            out_df = out_df.assign(c=c).rename({"c": i}, axis="columns")

    return out_df
####################################################
#
# For reasons to do with pandas' refusal to allow NaN
# in an `int` column, I'm going to return these as 
# dictionaries

def apply_theory(theory, *args, **kwargs):

    out_dict={r:[] for r in kwargs["responses_df"]["response_id"]}
    out_df = pd.DataFrame(index=kwargs["responses_df"]["response_id"])

    for i, rule in enumerate(theory):
        if rule["result"] in ["SATISFIABLE", "OPTIMUM FOUND"]:
            for r in evaluate_rule(rule=rule, **kwargs):
                out_dict[r].append(rule)

    return out_dict

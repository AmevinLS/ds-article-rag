import numpy as np
import pandas as pd

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import itertools
import argparse
from typing import List, Dict, Callable


def groups_to_dict(groups: List[List[int]]) -> Dict[int, int]:
    """Creates mapping from element to index of group it belongs to

    Args:
        groups (List[List[int]]): list of groups of elements

    Returns:
        Dict[int, int]: dictionary representing mapping element -> group_idx
    """
    elem_to_group = {}
    for i, group in enumerate(groups):
        for elem in group:
            elem_to_group[elem] = i
    return elem_to_group


def collapse_df_by_inds(
    df: pd.DataFrame, groups: List[List[int]], agg_mapping: Dict[str, Callable]
) -> pd.DataFrame:
    """Aggregates rows of dataframe within specified groups using functions specified in 'agg_mapping'

    Args:
        df (pd.DataFrame): dataframe, rows of which should be aggregated
        groups (List[List[int]]): list of groups of indices from 'df'
        agg_mapping (Dict[str, Callable]): dictionary mapping (column_name -> agg_function)

    Returns:
        pd.DataFrame: result of aggregating within groups
    """
    idx_to_group = groups_to_dict(groups)

    grouped = (
        df.groupby(pd.Series(df.index).apply(idx_to_group.__getitem__))
        .agg(agg_mapping)
        .reset_index(drop=True)
    )
    return grouped


def count_pattern_matches(texts: pd.Series, patterns: Dict[str, str]) -> pd.DataFrame:
    """For each text entry in a series, counts the number of matches for each pattern

    Args:
        texts (pd.Series): series of type 'str'
        patterns (Dict[str, str]): dictionary where keys are pattern names and values are regex string patterns

    Returns:
        pd.DataFrame: dataframe of shape (len(texts), len(patterns)) where column names are the pattern names given in 'patterns'
    """
    df_with_matches = pd.DataFrame()
    for patt_name, patt in patterns.items():
        df_with_matches[patt_name] = texts.str.count(patt)
    return df_with_matches


def group_by_consecutive_blocks(
    df: pd.DataFrame, indices: np.ndarray
) -> List[List[int]]:
    assert "article_idx" in df.columns

    is_in_indices_mask = np.zeros(len(df), dtype=np.bool_)
    is_in_indices_mask[indices] = True

    temp_df = df.copy()
    temp_df["in_indices"] = is_in_indices_mask
    groups = []
    for art_idx in temp_df["article_idx"].unique():
        article_df: pd.DataFrame = temp_df.loc[temp_df["article_idx"] == art_idx]
        for val, group in itertools.groupby(
            article_df.index, key=article_df["in_indices"].__getitem__
        ):
            if val:
                groups.append(list(group))
            else:
                groups.extend([[elem] for elem in group])
    return groups


def get_cos_sims(embeds: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeds, axis=1)
    return (embeds[1:] * embeds[:-1]).sum(axis=1) / norms[1:] / norms[:-1]


def group_by_consecutive_sims(
    df: pd.DataFrame, embeds: np.ndarray, std_multiplier=0.5
) -> List[List[int]]:
    assert "article_idx" in df.columns

    embed_sims = get_cos_sims(embeds)
    sim_mean, sim_std = embed_sims.mean(), embed_sims.std()

    groups = []
    for art_idx in df["article_idx"].unique():
        art_par_idxs = df.index[df["article_idx"] == art_idx].to_list()
        curr_group = [art_par_idxs[0]]
        for i in range(1, len(art_par_idxs)):
            if embed_sims[i - 1] > (sim_mean + std_multiplier * sim_std):
                curr_group.append(art_par_idxs[i])
            else:
                groups.append(curr_group)
                curr_group = [art_par_idxs[i]]
        if len(curr_group) > 0:
            groups.append(curr_group)

    return groups


def calc_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    embeds = np.array(hf_embeddings.embed_documents(texts))
    return embeds


def preprocess_complete(
    data_dir_path: str, join_std_multiplier: float = 0.5, verbose: bool = True
):
    # Loading and splitting data into paragraphs
    if verbose:
        print("Loading and splitting articles...")
    data = pd.read_csv(f"{data_dir_path}/medium.csv")
    paragraphs = data["Text"].str.split("\n\n").explode()
    pars_df = paragraphs.reset_index().rename(columns={"index": "article_idx"})

    # Detecting and joining 'code' paragraphs
    if verbose:
        print("Detecting and joining code paragraphs...")
    patterns = {
        "call": r"^[\w.]+\(",
        "getitem": r"^[\w.]+\[",
        "import": r"(^import |^from [\w.]+ import)",
        "class_def": r"^class \w+:$",
        "func_def": r"^def \w+\(",
        "return": r"^(return|yield|raise) ",
        "exception": r"^(try|catch|finally) .+:",
        "comment": r"#+ \w+",
        "loop": r"^(for|while) .+:",
        "if": r"^(if|else) .+:",
        "assign": r"^[\w.]+ ?[+\-\/*]?= ?.+",
        "no_letters": r"^[^A-Za-z]*$",
    }
    match_counts = count_pattern_matches(pars_df["Text"], patterns=patterns)
    code_suspects = pars_df.loc[match_counts.sum(axis=1) > 0]
    code_groups = group_by_consecutive_blocks(pars_df, code_suspects.index.to_numpy())
    reduced_pars_df = collapse_df_by_inds(
        pars_df,
        code_groups,
        agg_mapping={"article_idx": lambda idxs: idxs.mode(), "Text": "\n".join},
    )

    # Calculating paragraph embeddings and joining consecutive similar ones
    if verbose:
        print("Calculating embeddings and joining similar ones...")
    # TODO: Load saved embeddings if present in directory
    par_embeds = calc_embeddings(
        texts=reduced_pars_df["Text"].to_list(), model_name="all-MiniLM-L6-v2"
    )
    groups = group_by_consecutive_sims(
        reduced_pars_df, par_embeds, std_multiplier=join_std_multiplier
    )
    grouped_pars = collapse_df_by_inds(
        reduced_pars_df.loc[:, ["article_idx", "Text"]],
        groups,
        agg_mapping={
            "article_idx": lambda idxs: idxs.mode(),
            "Text": lambda texts: "\n\n".join(texts),
        },
    )

    # Adding metadata to final dataframe
    if verbose:
        print("Adding metadata to final dataframe...")
    grouped_pars = grouped_pars.merge(
        data["Title"], left_on="article_idx", right_on=data.index
    )
    grouped_pars = grouped_pars.reset_index(names="paragraph_idx")

    # Persisting results to disk
    if verbose:
        print("Saving to disk...")
    code_suspects.to_csv(f"{data_dir_path}/code_suspects.csv")
    reduced_pars_df.to_csv(f"{data_dir_path}/code_reduced_paragraphs.csv", index=False)
    grouped_pars.to_csv(f"{data_dir_path}/final_joined_paragraphs.csv", index=False)
    np.save(f"{data_dir_path}/code_reduced_par_embeds.npy", par_embeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="path to directory where 'medium.csv' resides"
    )
    parser.add_argument(
        "--join_std_mult",
        type=float,
        default=0.5,
        required=False,
        help="number of standard deviations to adjust the paragraph join threshold by",
    )

    args = parser.parse_args().__dict__
    preprocess_complete(args["data_dir"], join_std_multiplier=args["join_std_mult"])

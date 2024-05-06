import numpy as np
import pandas as pd

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import itertools
import os
import argparse
import csv
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
    """Groups DataFrame rows with indices into consecutive blocks.

    This function assumes the DataFrame has a column named "article_idx". It identifies rows with indices in 'indices'
    and groups them into consecutive blocks based on their article indices.

    Args:
        df (pd.DataFrame): The DataFrame containing the rows to be grouped.
        indices (np.ndarray): A NumPy array of indices of rows to be considered for grouping.

    Returns:
        List[List[int]]: A list of lists, where each inner list represents a group of consecutive row indices
        belonging to the same article.
    """
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
    """Calculates cosine similarities between consecutive embeddings in a NumPy array.

    Args:
        embeds (np.ndarray): A NumPy array of word embeddings.

    Returns:
        np.ndarray: A NumPy array of cosine similarities between consecutive embeddings in 'embeds'.
    """

    norms = np.linalg.norm(embeds, axis=1)
    return (embeds[1:] * embeds[:-1]).sum(axis=1) / norms[1:] / norms[:-1]


def group_by_consecutive_sims(
    df: pd.DataFrame, embeds: np.ndarray, std_multiplier=0.5
) -> List[List[int]]:
    """Groups DataFrame rows with indices into consecutive blocks based on cosine similarity of embeddings.

    This function assumes the DataFrame has a column named "article_idx". It calculates cosine similarities between
    consecutive word embeddings in 'embeds' and uses them to group rows in 'df' with similar embeddings into
    consecutive blocks based on their article indices.

    Args:
        df (pd.DataFrame): The DataFrame containing the rows to be grouped.
        embeds (np.ndarray): A NumPy array of word embeddings corresponding to rows in 'df'.
        std_multiplier (float, optional): A multiplier to adjust the threshold for grouping based on standard
            deviation of cosine similarities. Defaults to 0.5.

    Returns:
        List[List[int]]: A list of lists, where each inner list represents a group of consecutive row indices
        belonging to the same article.
    """

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
    """Calculates word embeddings for a list of strings using a sentence_transformers model.

    Args:
        texts (List[str]): A list of strings for which word embeddings need to be calculated.
        model_name (str): The name of the sentence_transformers model to be used for embedding calculation.

    Returns:
        np.ndarray: A NumPy array of word embeddings, where each row represents the embedding of a corresponding text in 'texts'.
    """

    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    embeds = np.array(hf_embeddings.embed_documents(texts))
    return embeds


def preprocess(
    data_dir_path: str, join_std_multiplier: float = 0.5, verbose: bool = True
):
    """Preprocesses directory containing 'medium.csv' dataset file and generates a DataFrame of joined paragraphs.

    This function performs the following steps:

    1. Loads and splits articles into paragraphs from 'medium.csv'.
    2. Detects and joins 'code' paragraphs based on regular expression patterns.
    3. Calculates word embeddings for the paragraphs or load them if cache is present.
    4. Groups consecutive paragraphs with similar embeddings into blocks.
    5. Joins the text content of paragraphs within each block.
    6. Merges article titles from 'medium.csv' and adds a 'paragraph_idx' column for indexing.

    Args:
        data_dir_path (str): The path to the directory containing 'medium.csv' file.
        join_std_multiplier (float, optional): A multiplier to adjust the threshold for grouping paragraphs
            based on standard deviation of cosine similarities. Defaults to 0.5.
        verbose (bool, optional): A flag to control whether to print progress messages. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing preprocessed data with columns 'article_idx', 'Text', and 'Title'.
            'article_idx' groups paragraphs belonging to the same article, 'Text' contains the joined text content
            of paragraphs within a block, and 'Title' is the title of the corresponding article.
    """

    cache_dir_path = os.path.join(data_dir_path, "cache")
    if not os.path.exists(cache_dir_path):
        os.mkdir(cache_dir_path)
    code_reduced_df_path = os.path.join(cache_dir_path, "code_reduced_paragraphs.csv")
    code_reduced_embeds_path = os.path.join(
        cache_dir_path, "code_reduced_par_embeds.npy"
    )

    # Loading and splitting data into paragraphs
    if verbose:
        print("Loading and splitting articles...")
    data = pd.read_csv(os.path.join(data_dir_path, "medium.csv"))
    paragraphs = data["Text"].str.split("\n\n").explode()
    pars_df = paragraphs.reset_index().rename(columns={"index": "article_idx"})

    # Detecting and joining 'code' paragraphs
    if os.path.exists(code_reduced_df_path) and os.path.exists(
        code_reduced_embeds_path
    ):
        reduced_pars_df = pd.read_csv(code_reduced_df_path, keep_default_na=False)
        par_embeds: np.ndarray = np.load(code_reduced_embeds_path)
        assert reduced_pars_df.shape[0] == par_embeds.shape[0]
    else:
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
        code_groups = group_by_consecutive_blocks(
            pars_df, code_suspects.index.to_numpy()
        )
        reduced_pars_df = collapse_df_by_inds(
            pars_df,
            code_groups,
            agg_mapping={"article_idx": lambda idxs: idxs.mode(), "Text": "\n".join},
        )
        reduced_pars_df.to_csv(code_reduced_df_path, quoting=csv.QUOTE_ALL, index=False)

        if verbose:
            print("Calculating embeddins...")
        par_embeds = calc_embeddings(
            texts=reduced_pars_df["Text"].to_list(), model_name="all-MiniLM-L6-v2"
        )
        np.save(code_reduced_embeds_path, par_embeds)

    # Joining consecutive similar paragraphs
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

    return grouped_pars


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
    grouped_pars = preprocess(
        args["data_dir"], join_std_multiplier=args["join_std_mult"]
    )
    grouped_pars.to_csv(
        os.path.join(args["data_dir"], "final_joined_paragraphs.csv"),
        quoting=csv.QUOTE_ALL,
    )

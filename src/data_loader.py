#MODULE-1
#src/data_loader.py
#PURPOSE: Load all five datasets, unify schemas, produce three output dataframes.
#OUTPUTS: df_master(binary classification), df_tele(regression),df_istanbul(holdout).

import pandas as pd
import numpy as np
import yaml
import os
import re
from pathlib import Path


#Load the project configuration file so every path is centralised
cfg = yaml.safe_load(open("config.yaml"))


def load_oxford() -> pd.DataFrame:
    """
    Load UCI Oxford Parkinson's dataset (parkinsons.csv)
    197 rows, 22 columns. Cotains 'name' (subject ID encoded).
    22 acoustic features, and 'status' (1=PD, 0=Healthy).
    """
    #Read the CSV from the path defined in config
    df = pd.read_csv(cfg["files"]["oxford"])

    #Extract subject ID from the 'name' column.
    #Format: "phon_R01_S01_6" ->wew want "phon_R01_S01"
    #This is critical for subject-level splitting in Stage 4.
    df["subject_id"] = df["name"].apply(
        lambda x: "_".join(str(x).split("_")[:3])
    )

    #Rename the target column to a unified name 'label'
    df = df.rename(columns={"status":"label"})

    #Add a source column to track datset origin after concatenation
    df["source"]="oxford"

    #Drop the original name column - subject_id replaces it
    df = df.drop(columns=["name"])

    print(f"Oxford loaded; {df.shape},PD={df.label.sum()}, Healthy={(df.label==0).sum()}")

    return df

def load_replicated() ->pd.DataFrame:
    """
    Load UCI Replicated Features dataset (pd_speech_features.csv).
    240 rows from 80 subjects (3 recordings each). 753 features including MFCCs.
    Target column is 'class' (1=PD, 0=Healthy). Perfectly balanced: 40 PD, 40 healthy.
    """
    df = pd.read_csv(cfg["files"]["replicated"])


    #Rename target to unified name
    df =df.rename(columns={"class":"label"})

    #Replicated dataset does not have a subject_id_column.
    #We create a proxy: row index divided by 3 (since 3 recs pere subject)
    # df["subject_id"] = (df.index//3).astype(str).apply(lambda x: f"rep_subj{x}")
    df["subject_id"] = "rep_subj"+ (df.index//3).astype(str)
    df["source"] ="replicated"
    print(f"Replicated loaded: {df.shape}, PD={df.label.sum()},Healthy={(df.label==0).sum()}")
    return df

def load_telemonitoring() -> pd.DataFrame:
    """
    Load Parkinson's Telemonitoring Dataset (parkinsons_updrs.data)
    5,875 rows from 42 subjects. Multiple recordings per subject over time.
    Targets: motor_UPDRS and total_UPDRS (continuous - regression task.)
    Do not merge with df_master.
    """
    df = pd.read_csv(cfg["files"]["tele"])

    #standard subject column name for GroupKFold splitting
    df =df.rename(columns={"subject":"subject_id"})
    df["subject_id"] = "tele_subj_" + df["sunject_id"].astype(str)
    print(f"Telemonitoring loaded: {df.shape}")
    print(f"motor_UPDRS range: {df.motor_UPDRS.min():.1f}-{df.motor_UPDRS.max():.1f}")
    print(f"total_UPDRS range: {df.total_UPDRS.min():.1f}-{df.total_UPDRS.max():.1f}")
    return df


def load_istanbul() ->pd.DataFrame:
    """
    Load Istanbul PD Speech features.
    252 subjects, 26 acoustic features.
    ###SEALED - DO NOT USE UNTIL STAGE 9 CROSS-CORPUS EVALUTAION###
    This dataset is never split, never used in training, never used for tuning.
    """
    df = pd.read_csv(cfg["files"]["istanbul"])

    #Identify the target column - may differ from 'label'
    #Common names in this dataset: 'class', 'status'
    label_col = [c for c in df.columns if c.lower() in ['class','status','label']][0]
    df = df.rename(columns={label_col:"label"})
    df["source"] = "istanbul"

    #IMPORTANT: no subject_id in Istanbul - each row is one subject
    df["subject_id"] = "ist_subj_"+df.index.astype(str)
    print(f"Istanbul loaded: {df.shape} ###SEALED-do not touch until Stage 9###")
    return df

def build_master_frame(df_oxford:pd.DataFrame, df_replicated: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Oxford and Replicated on the 8 shared acoustic features.
    These 8 features exist in both datasets adn form the joint feature space.
    Fianl output: - 435rows*11columns(8 features + label + source + subject_id).
    """
    # The 8 features shared across Oxford, Replicated and Istanbul
    shared = cfg["features"]["shared_8"]

    #Select only the shared features + metadata columns from each dataset
    cols = shared + ["label","source","subject_id"]
    
    #Keep only columns that exist in each dataframe
    oxford_cols = [c for c in cols if c in df_oxford.columns]
    rep_cols = [c for c in cols if c in df_replicated.columns]
    df_o = df_oxford[oxford_cols].copy()
    df_r = df_replicated[rep_cols].copy()

    # Ensure both dataframes have all 8 shared features (fill missing with NaN)
    for feat in shared:
        if feat not in df_o.columns: df_o[feat] = np.nan
        if feat not in df_r.columns: df_r[feat] = np.nan

    # concatenate vertically - rows from both datasets
    df_master = pd. concat([df_o[cols],df_r[cols]], ignore_index=True)
    
    print(f"\ndf_master built: {df_master.shape}")
    print(f"PD: {df_master.label.sum()}, Healthy:{(df_master.label==0).sum()}")
    print(f"Sources: {df_master.source.value_counts().to_dict()}")
    print(f"Missing values;\n{df_master.isnull().sum()}")
    return df_master

def run_ingestion() -> tuple:
    """
    Main entry point, Runs the full Stage 1 pipeline.
    Returns: (df_master, df_tele, df_istanbul)
    """
    print("="*60)
    print("STAGE 1: Data Ingestion & Schema Unification")
    print("="*60)

    #Load all tabular datasets
    df_ox = load_oxford()
    df_rep = load_replicated()
    df_tele =load_telemonitoring()

    #Load Istanbul and immediately seal it
    df_istanbul = load_istanbul()
    #============================================================
    #DO NOT TOUCH df_istanbul UNTIL STAGE 9
    #============================================================


    #Build the master binary classification frame
    df_master = build_master_frame(df_ox,df_rep)


    #Save processed fames for downstream stages
    Path(cfg["paths"]["processed"]).mkdir(parents=True, exist_ok=True)
    df_master.to_csv(f'{cfg["paths"]["processed"]}master_raw.csv',index=False)
    df_tele.to_csv(f"{cfg["paths"]["processed"]}tele_raw.csv",index=False)
    df_istanbul.to_csv(f"{cfg["path"]["processed"]}istanbul_raw.csv",index=False)

    print("\nStage 1 complete. All frames saved to data/processed/")
    return df_master, df_tele, df_istanbul


if __name__ == "__main__":
    df_master, df_tele, df_istanbul = run_ingestion()
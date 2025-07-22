"""
Preprocessing / Pré-processamento

This module contains functions for preprocessing the Shinkansen Travel Experience dataset.

---
It includes functions for applying one-hot encoding to categorical variables, handling missing values, and scaling numerical features.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

###################################################################################################
### Random Forest Preprocessing Functions
###################################################################################################

def one_hot_encode_fit_transform(df, target_col=None):
    """
    Faz fit e transform do OneHotEncoder em df.
    Detecta automaticamente colunas categóricas.
    
    Retorna:
        encoder -> OneHotEncoder treinado
        X_transformed -> DataFrame transformado (sem target)
        y -> Series target (ou None)
    """
    # Se tiver target_col, separa
    if target_col is not None:
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
    else:
        X = df.copy()
        y = None

    # Detecta colunas categóricas automaticamente
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        print("No categorical columns found for one-hot encoding.")
        return None, X, y

    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False  # Use sparse_output=False to return a dense array
    )
    encoder.fit(X[cat_cols])
    
    cat_transformed = encoder.transform(X[cat_cols])
    cat_columns = encoder.get_feature_names_out()
    df_cat = pd.DataFrame(cat_transformed, columns=cat_columns, index=X.index)
    
    df_num = X.drop(columns=cat_cols)
    X_transformed = pd.concat([df_num, df_cat], axis=1)
    
    return encoder, X_transformed, y

def one_hot_encode_transform(df, encoder, target_col=None, train_columns=None):
    """
    Transforma novos dados com um encoder treinado.
    Alinha colunas com treino, criando as faltantes como zero,
    e descartando as extras.
    
    Retorna:
        X_transformed -> DataFrame transformado (sem target)
        y -> Series target (ou None)
    """
    if target_col is not None:
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
    else:
        X = df.copy()
        y = None

    # Detecta as colunas categóricas automaticamente
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        print("No categorical columns found for one-hot encoding.")
        X_transformed = X.copy()
    else:
        cat_transformed = encoder.transform(X[cat_cols])
        cat_columns = encoder.get_feature_names_out()
        df_cat = pd.DataFrame(cat_transformed, columns=cat_columns, index=X.index)
        df_num = X.drop(columns=cat_cols)
        X_transformed = pd.concat([df_num, df_cat], axis=1)

    # Se informado train_columns, alinhar as colunas:
    if train_columns is not None:
        # adiciona colunas faltantes
        missing_cols = set(train_columns) - set(X_transformed.columns)
        for col in missing_cols:
            X_transformed[col] = 0
        
        # remove colunas extras
        extra_cols = set(X_transformed.columns) - set(train_columns)
        if extra_cols:
            X_transformed.drop(columns=list(extra_cols), inplace=True)
        
        # reorganiza ordem
        X_transformed = X_transformed[train_columns]

    return X_transformed, y

def apply_one_hot_encoding(df, target_col='status'):
    """
    (PT-BR)
    Aplica one-hot encoding nas colunas categóricas do DataFrame.

    (EN-US)
    Apply one-hot encoding to categorical columns in the DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with categorical columns.
        target_col (str): Name of the target column to be excluded from encoding.
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical columns.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        print("No categorical columns found for one-hot encoding.")
        return df
    
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    df_encoded = pd.concat([X_encoded, y.reset_index(drop=True)], axis=1)

    return df_encoded

if __name__ == "__main__":
    pass
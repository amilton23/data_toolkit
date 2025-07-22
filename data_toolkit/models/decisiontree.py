"""
This module contains functions for training and optimizing decision tree-based models.

---
It includes implementations for LightGBM, Random Forest and catboost classifiers, as well as hyperparameter tuning using Optuna.
It also provides functions for plotting feature importances and decision trees.
"""

import pandas as pd
import numpy as np  

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

import lightgbm as lgb
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from catboost import CatBoostClassifier, Pool, cv, CatBoostError

# Configurações de exibição do pandas
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 200)
# Configurações do matplotlib
plt.rcParams['figure.figsize'] = (10, 6)

import optuna
warnings.filterwarnings("ignore")


###################################################################################################
### Random Forest
################################################################################################### 

def train_random_forest(df, target_col, n_top_feat_importance = 20, test_size=0.2, random_state=42, params=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': random_state,
            'n_jobs': -1
        }
    else:
        params['random_state'] = random_state

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Classification report:\n", report)

    # Importância das features
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:{n_top_feat_importance}]

    print(f"Top {n_top_feat_importance} features (impurity-based):")
    for feat, imp in feat_imp:
        print(f"{feat}: {imp:.4f}")

    # Plotar
    top_features = [f[0] for f in feat_imp]
    top_importances = [f[1] for f in feat_imp]

    plt.figure(figsize=(10,6))
    sns.barplot(x=top_importances[::-1], y=top_features[::-1], palette="viridis")
    plt.xlabel('Importance')
    plt.title(f'Top {n_top_feat_importance} Features - Random Forest')
    plt.show()

    return model

def optimize_random_forest(df, target_col='status', n_trials=100, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': random_state,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best Trial:")
    print(f"Accuracy: {study.best_value:.4f}")
    print("Best Params:", study.best_params)

    return study.best_params

def plot_random_forest_tree(model, feature_names, class_names=None, tree_index=0, max_depth=5):
    """    Plots a specific tree from a Random Forest model.
    Arguments:
        model (RandomForestClassifier): The trained Random Forest model.
        feature_names (list): List of feature names.
        class_names (list, optional): List of class names for classification tasks.
        tree_index (int, optional): Index of the tree to plot. Default is 0.
        max_depth (int, optional): Maximum depth of the tree to plot. Default is 5.
    Returns:
        None: Displays the plot of the specified tree.
    """
    estimator = model.estimators_[tree_index]

    plt.figure(figsize=(20, 10))
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10
    )
    plt.title(f"Tree {tree_index} of Random Forest")
    plt.show()

###################################################################################################
### LightGBM
################################################################################################### 

def train_lightgbm(df, target_col='status', test_size=0.2, random_state=42, params=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar colunas categóricas do tipo object
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Converter colunas categóricas para category
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # Dividir treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Criar datasets LightGBM com categorical_feature
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain, categorical_feature=categorical_cols)

    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'shrinkage_rate': 0.12,
            'num_leaves': 25,
            'max_depth': 4,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'random_state': random_state
        }
    else:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': random_state,
            **params
        }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dvalid]
    )

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary)

    print(f"Accuracy: {acc:.4f}")
    print("Classification report:\n", report)

    importances = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()

    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]

    print("Top 20 features (gain):")
    for feat, imp in feat_imp:
        print(f"{feat}: {imp}")

    top_features = [f[0] for f in feat_imp]
    top_importances = [f[1] for f in feat_imp]

    plt.figure(figsize=(10,6))
    plt.barh(top_features[::-1], top_importances[::-1], color='skyblue')
    plt.xlabel('Importance (gain)')
    plt.title('Top 20 Features by Gain')
    plt.show()

    ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain'])
    plt.show()

    return model

def optimize_lightgbm(df, target_col='status', test_size=0.2, random_state=42, n_trials=50):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            # 'shrinkage_rate': trial.suggest_float('shrinkage_rate', 0.01, 1.0),
            # 'eta': trial.suggest_float('eta', 0.01, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'random_state': random_state
        }

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid]
        )

        preds = model.predict(X_valid)
        preds_binary = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_valid, preds_binary)

        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters found:")
    print(study.best_params)
    print(f"Best Accuracy: {study.best_value:.4f}")

    # Treinar o modelo final com os melhores hiperparâmetros no dataset inteiro
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'random_state': random_state
    })

    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(best_params, dtrain_full, num_boost_round=100)

    return final_model, study.best_params

###################################################################################################
### Catboost 
################################################################################################### 

def train_catboost(df, target_col='status', test_size=0.2, random_state=42, params=None):
    """
    Treina CatBoost em dados tabulares, mostra métricas e importância de variáveis.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar variáveis categóricas (CatBoost faz isso automaticamente, mas informá-las ajuda)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Codificar categóricas como category se não estiverem
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if params is None:
        params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 4,
            'eval_metric': 'Accuracy',
            'random_seed': random_state,
            'verbose': 100,
            'loss_function': 'Logloss'
        }
    else:
        # garantir parâmetros mínimos necessários
        params.update({
            'random_seed': random_state,
            'eval_metric': 'Accuracy',
            'verbose': 100,
            'loss_function': 'Logloss'
        })

    # CatBoost Pool para treino
    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)

    # Importância das variáveis
    feature_importances = model.get_feature_importance(prettified=True)
    feature_importances = feature_importances.sort_values("Importances", ascending=False).head(20)

    print("Top 20 Features:")
    print(feature_importances)

    # Plotar as importâncias
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importances', y='Feature Id', data=feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features - CatBoost')
    plt.show()

    return model

def optimize_catboost(df, target_col='status', test_size=0.2, random_state=42, n_trials=50):
    """
    Otimiza hiperparâmetros do CatBoost usando Optuna.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # garantir types
    for col in categorical_cols:
        X[col] = X[col].astype('category')

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_seed': random_state,
            'loss_function': 'Logloss',
            'eval_metric': 'Accuracy',
            'verbose': 0
        }

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        valid_pool = Pool(X_valid, y_valid, cat_features=categorical_cols)

        try:
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50, verbose=0)

            preds = model.predict(X_valid)
            acc = accuracy_score(y_valid, preds)
            return acc

        except CatBoostError as e:
            print("CatBoostError:", e)
            return 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters found:")
    print(study.best_params)
    print(f"Best Accuracy: {study.best_value:.4f}")

    # Treinar modelo final no dataset inteiro
    best_params = study.best_params
    best_params.update({
        'random_seed': random_state,
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'verbose': 100
    })

    train_pool_full = Pool(X, y, cat_features=categorical_cols)
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(train_pool_full)

    return final_model, study.best_params
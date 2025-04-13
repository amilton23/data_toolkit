"""
Package de uso de dados e arquivos. 

"""

import datetime
from datetime import datetime, timezone
from typing import TypeVar

import numpy as np
import pandas as pd
import re 
import os 
import json
import re

T = TypeVar("T")

def load_local_key(keys_path, key_name):
    """
    Caso não esteja utilizando credenciais em repositório.
    """
    if os.path.exists(keys_path):
        with open(keys_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            if key_name not in data or data[key_name] is None or data[key_name] == "":
                print(f"Chave {key_name} não encontrada ou vazia.")
                return None
            return data[key_name]
    return None

def create_dir_returning_path(name:str):
    """
    Cria pasta de nome passado como string. Função utilitária de model_evaluation_cm().
    
    Args:
        name (str): nome da pasta em formato de string
        
    Return:
        path (str): caminho da pasta em string para utilização desta
    """
    path_dir = os.getcwd() + f'/{name}/'

    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    
    return path_dir

def switch_label(dict_mapping, series):
    """
    Altera label string para resposta quantitativa a partir de dict, saída é uma pandas Series.
    """
    # Inverter o dicionário para ter as chaves como valores e vice-versa
    #inverted_dict = {v: k for k, v in dict_mapping.items()}
    
    # Mapear os rótulos usando o dicionário invertido
    result_series = series.map(dict_mapping)
    
    # Substituir os valores não mapeados por NaN
    result_series = result_series.fillna(series)    
    
    return result_series

def get_current_datetime():
    return datetime.now(timezone.utc)


def load_data(file_path, sheet_name = 1):
    """
    
    """
    # Carrega os dados do arquivo
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)

    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path, sheet_name = sheet_name)

    elif file_path.endswith(".parquet"):
        data = pd.read_parquet(file_path)
    
    else:
        data = None
        print("ERRO! O arquivo que deseja ler não é csv, excel ou parquet.")
    return data

def keep_number(text):
    text = re.sub(r'[^0-9]', '', str(text))
    return text

def map_values(value, mapping_dict, reverse=False):
    """
    Mapeia um valor para o respectivo valor numérico ou de string com base em um dicionário de mapeamento.

    Parâmetros:
    - value: Valor a ser mapeado.
    - mapping_dict: Dicionário de mapeamento {'valor_original': valor_alvo}.
    - reverse: Se True, realiza a conversão de valor numérico para string; caso contrário, converte de string para numérico.

    Retorna:
    - Valor correspondente ao valor original no dicionário de mapeamento.
    """
    if reverse:
        reverse_mapping = {v: k for k, v in mapping_dict.items()}
        return reverse_mapping.get(value, value)
    else:
        return mapping_dict.get(value, value)
    
def text_preprocessing(text : str):
    """
    Deixa o texto em tudo minúsculo e remove caracteres especiais mantendo letras/vogais e espaços entre termos.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]*\s*$', '', text)
    return text

def deslocar_para_esquerda(df, linha_idx, coluna_inicio):
    """
    Desloca os valores de uma linha do DataFrame uma posição para a esquerda,
    preenchendo com None nas colunas à direita da coluna selecionada.

    Parâmetros:
    - df: DataFrame do pandas.
    - linha_idx: Índice da linha a ser deslocada.
    - coluna_inicio: Nome da coluna a partir da qual os valores serão deslocados para a esquerda.

    Retorna:
    - DataFrame com a linha deslocada uma posição para a esquerda e preenchida com None nas colunas à direita.
    """

    # Criar uma cópia do DataFrame para evitar alterações indesejadas
    df_copia = df.copy()

    # Obter o índice numérico da coluna a ser utilizada como referência
    coluna_inicio_idx = df_copia.columns.get_loc(coluna_inicio)

    # Salvar o valor da coluna de referência
    valor_coluna_referencia = df_copia.iloc[linha_idx, coluna_inicio_idx]

    # Deslocar os valores para a esquerda
    df_copia.iloc[linha_idx, coluna_inicio_idx:-1] = df_copia.iloc[linha_idx, coluna_inicio_idx+1:].values

    # Inserir o valor da coluna de referência na última posição
    df_copia.iloc[linha_idx, -1] = valor_coluna_referencia

    return df_copia

def concatenar_valores_em_linhas(df, coluna_origem, coluna_destino, linhas_indices):
    """
    Adiciona os valores de uma coluna apenas em linhas específicas, concatenando com o que já está
    presente na coluna de destino, e armazena o resultado na mesma coluna.

    Parâmetros:
    - df: DataFrame do pandas.
    - coluna_origem: Nome da coluna de origem.
    - coluna_destino: Nome da coluna de destino.
    - linhas_indices: Lista de índices das linhas específicas onde a operação deve ser realizada.

    Retorna:
    - DataFrame com os valores adicionados nas linhas específicas na coluna de destino.
    """

    # Verifica se a coluna de origem existe no DataFrame
    if coluna_origem not in df.columns:
        print(f"A coluna '{coluna_origem}' não existe no DataFrame.")
        return df

    # Concatena os valores da coluna de origem com o que já está presente nas linhas específicas da coluna de destino
    df.loc[df.index.isin(linhas_indices), coluna_destino] += " "
    df.loc[df.index.isin(linhas_indices), coluna_destino] += df.loc[df.index.isin(linhas_indices), coluna_origem].astype(str).apply(lambda x: ' '.join(x.split() if isinstance(x, str) else []))

    return df

def transform_and_insert(valor, col_name = 'col'):
    """
    Remove pontos e substitui vírgulas por ponto, depois divide a string pelos espaços e converte cada parte para float, retornando uma tupla.

    Parâmetros:
    - valor = string com a formatação em string escolhida.

    Retorno:
    - valores = pd.series de valores em type float.
    """
    if isinstance(valor, str):
        valor = valor.replace('.', '').replace(',', '.')
        valores = [float(part) for part in valor.split('a')]
        return pd.Series(valores, index=[f'{col_name}_min', f'{col_name}_max'])
    
    elif isinstance(valor, float):
        return pd.Series([valor, valor], index=[f'{col_name}_min', f'{col_name}_max'])
    
    else:
        raise ValueError(f"Tipo de valor não suportado: {type(valor)}")



if __name__ == "__main__":
    pass

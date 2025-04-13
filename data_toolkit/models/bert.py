"""
(PT-BR)
Modelo BERT para classificação de texto utilizando a biblioteca HuggingFace Transformers.

(EN-US)
BERT model for text classification using the HuggingFace Transformers library.
"""

from time import time
from typing import Callable, List, Optional

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from ..utils.dataviz import print_progress_bar
from ..utils.general import get_current_datetime

class BERTHuggingFaceClassification:
    """

    (PT-BR)
    Descrição:
        Classe que implementa o modelo BERT para classificação de texto utilizando a biblioteca HuggingFace Transformers.
        O modelo é inicializado com o tokenizador e o modelo pré-treinado do BERT. O treinamento pode ser realizado com K-Fold Cross Validation ou com um conjunto de treinamento e teste. A inferência pode ser realizada em um DataFrame do pandas.
        O modelo pode ser salvo e carregado a partir de um arquivo .pth.
        O modelo é treinado utilizando o otimizador AdamW e a função de perda CrossEntropyLoss.
        O modelo é treinado em GPU se disponível, caso contrário, em CPU.

    Atributos:
        X (pd.Series): Series do pandas contendo os textos.
        y (pd.Series): Series do pandas contendo as labels no formato integer (0, 1, ...)
        X_col_nm (str): Nome da coluna que contém os textos que serão classificados pelo BERT.
        y_infer_col_nm (str): Nome da coluna que contém as labels inferidas pelo BERT.
        
        tokenizer (BertTokenizer): Tokenizer do BERT.
        model (BertForSequenceClassification): Modelo BERT para classificação de texto.
        device (torch.device): Dispositivo utilizado para treinamento e inferência (GPU ou CPU).
        model_name (str): Nome do modelo BERT pré-treinado utilizado.
        _dict_labels (dict): Dicionário de labels para as classes do modelo.
        _preprocess (Callable): Função de pré-processamento dos textos.
        threshold (float): Limite de probabilidade para considerar uma classe como positiva.
        foundation_model (str): Nome do modelo BERT pré-treinado utilizado.
        preprocess (Callable): Função de pré-processamento dos textos.
        dict_labels (dict): Dicionário de labels para as classes do modelo.
        size (str): Tamanho do modelo BERT utilizado ("base" ou "large").
        epochs (int): Número de epochs utilizados no treinamento.
        batch_size (int): Tamanho do lote utilizado no treinamento.
        test_size (float): Proporção utilizado para teste para validação pós treinamento.
        random_state (int): Semente do gerador de números aleatórios utilizado para separação dos conjuntos de treinamento e teste.
        save_model (bool): Condição para salvar modelo.
        model_dir (str): Nome do modelo salvo.
        prob_column_name (str): Nome da coluna que contém as probabilidades das labels inferidas pelo BERT.
        k_folds (int): Número de folds para Cross Validation.
        random_state (int): Semente do gerador de números aleatórios para reprodutibilidade.
        save_model (bool): Condição para salvar o modelo.
        model_dir (str): Caminho para salvar o modelo.
    
    (EN-US)
    Description:
        Class that implements the BERT model for text classification using the HuggingFace Transformers library.
        The model is initialized with the tokenizer and the pre-trained BERT model. Training can be performed with K-Fold Cross Validation or with a training and test set. Inference can be performed on a pandas DataFrame.
        The model can be saved and loaded from a .pth file.
        The model is trained using the AdamW optimizer and the CrossEntropyLoss function.
        The model is trained on GPU if available, otherwise on CPU.

    Attributes:
        X (pd.Series): Pandas Series containing the texts.
        y (pd.Series): Pandas Series containing the labels in integer format (0, 1, ...).
        X_col_nm (str): Name of the column containing the texts to be classified by BERT.
        y_infer_col_nm (str): Name of the column containing the labels inferred by BERT.
        
        tokenizer (BertTokenizer): BERT tokenizer.
        model (BertForSequenceClassification): BERT model for text classification.
        device (torch.device): Device used for training and inference (GPU or CPU).
        model_name (str): Name of the pre-trained BERT model used.
        _dict_labels (dict): Dictionary of labels for the model's classes.
        _preprocess (Callable): Function for text preprocessing.
        threshold (float): Probability threshold to consider a class as positive.
        foundation_model (str): Name of the pre-trained BERT model used.
        preprocess (Callable): Function for text preprocessing.
        dict_labels (dict): Dictionary of labels for the model's classes.
        size (str): Size of the BERT model used ("base" or "large").
        epochs (int): Number of epochs used in training.
        batch_size (int): Batch size used in training.
        test_size (float): Proportion used for testing for post-training validation.
        random_state (int): Random seed used for splitting training and testing sets.
        save_model (bool): Condition to save the model.
        model_dir (str): Name of the saved model.
        prob_column_name (str): Name of the column containing the probabilities of the labels inferred by BERT.
        k_folds (int): Number of folds for Cross Validation.
        random_state (int): Random seed for reproducibility.
        save_model (bool): Condition to save the model.
        model_dir (str): Path to save the model.
    """

    tokenizer: BertTokenizer = None
    model: BertForSequenceClassification = None
    device: torch.device = None

    def __init__(
        self,
        size: str = "base",
        dict_labels=None,
        preprocess=None,
        foundation_model=None,
        threshold=None,
    ):
        """
        (PT-BR)
        Descrição:
            Inicializa o modelo BERT para classificação de texto.
        Args:
            size (str, optional): "base" para  versão com 12 camadas e 110M de parâmetros e "large" para a versão com 24 camadas e 335M de parâmetros. Defaults to "base".
            dict_labels (dict, optional): Dicionário de labels para as classes do modelo. Defaults to None.
            preprocess (Callable, optional): Função de pré-processamento dos textos. Defaults to None.
            foundation_model (str, optional): Nome do modelo BERT pré-treinado utilizado. Defaults to None.
            threshold (float, optional): Limite de probabilidade para considerar uma classe como positiva. Defaults to None.
        Retorna:
            None
        
        (EN-US)
        Description:
            Initializes the BERT model for text classification.
        Args:
            size (str, optional): "base" for the version with 12 layers and 110M parameters and "large" for the version with 24 layers and 335M parameters. Defaults to "base".
            dict_labels (dict, optional): Dictionary of labels for the model's classes. Defaults to None.
            preprocess (Callable, optional): Function for text preprocessing. Defaults to None.
            foundation_model (str, optional): Name of the pre-trained BERT model used. Defaults to None.
            threshold (float, optional): Probability threshold to consider a class as positive. Defaults to None.
        Returns:
            None
        """

        bert_config = {
            "base": {"layers": 12, "params": "110M"},
            "large": {"layers": 24, "params": "335M"},
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name: str = f"neuralmind/bert-{size}-portuguese-cased"
        
        self.threshold = threshold
        
        if foundation_model is not None:
            self.model_name = foundation_model

        print(f"BERT model initialized with '{self.model_name}' pretrained model")
        print("  Config:")
        print(f"    - {bert_config[size]['layers']} layers")
        print(f"    - {bert_config[size]['params']} parameters")
        print(f"    - set to {self.device}")
        print("")

        if dict_labels:
            self._dict_labels = dict_labels

            if "0" not in self._dict_labels:
                self._dict_labels["0"] = "NEGATIVE"
            if "1" not in self._dict_labels:
                self._dict_labels["1"] = "POSITIVE"
            if "-1" not in self._dict_labels:
                self._dict_labels["-1"] = "NOT FOUND"
            if "-2" not in self._dict_labels:
                self._dict_labels["-2"] = "UNDEFINED"
            if "-3" not in self._dict_labels:
                self._dict_labels["-3"] = "UNKNOWN"
            if "-4" not in self._dict_labels:
                self._dict_labels["-4"] = "EMPTY"
        else:
            self._dict_labels = {
                "0": "NEGATIVE",
                "1": "POSITIVE",
                "-1": "NOT FOUND",
                "-2": "UNDEFINED",
                "-3": "UNKNOWN",
                "-4": "EMPTY",
            }

        if preprocess is None:
            self._preprocess = lambda x: x
        else:
            self._preprocess = preprocess

    def preprocess(self, text):
        """
        (PT-BR)
        Realiza o pré-processamento do texto fornecido.

        (EN-US)
        Preprocesses the provided text.
        """

        text = self._preprocess(text)
        return text

    def train_k_fold(
        self,
        X: pd.Series,
        y: pd.Series,
        epochs: int = 5,
        batch_size: int = 8,
        k_folds: int = 5,
        random_state: int = 42,
        save_model: bool = True,
        model_dir: str = "bert_model.pth",
    ):
        """
        (PT-BR)
        Descrição:
            Função que executa o treinamento do modelo BERT com K-Fold Cross Validation.
        Args:
            X (pd.Series): Series do pandas contendo os textos
            y (pd.Series): Series do pandas contendo as labels no formato integer (0, 1, ...)
            epochs (int): Número de epochs utilizados no treinamento. Defaults to 5.
            batch_size (int): Tamanho do lote utilizado no treinamento. Defaults to 8.
            k_folds (int): Número de folds para Cross Validation. Defaults to 5.
            random_state (int): Semente do gerador de números aleatórios para reprodutibilidade. Defaults to 42.
            save_model (bool): Condição para salvar o modelo. Defaults to True.
            model_dir (str): Caminho para salvar o modelo. Defaults to "bert_model.pth".
        Retorna:
            model (BertForSequenceClassification): Modelo BERT treinado com K-Fold Cross Validation.
        """
        print("(PT-BR) Treinamento do modelo iniciado com K-Fold Cross Validation")
        print("(EN-US) Model training started with K-Fold Cross Validation")

        best_validation_loss = 1
        kfold_escolhido = 1
        epoch_escolhida = 1
        
        # Preparar o KFold
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

        # Converter os dados em arrays para facilitar o manuseio
        X = X.to_numpy()
        y = y.to_numpy()

        # Loop pelos folds
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            
            # Carregar o tokenizador e o modelo pré-treinado
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            
            print(f"\nFold {fold + 1}/{k_folds}")

            # Separar os índices para treino e validação
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Tokenizar os textos de treino e validação
            train_encodings = self.tokenizer(
                list(X_train),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            val_encodings = self.tokenizer(
                list(X_val),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            print(f"  - tokenização dos conjuntos de treino e validação")

            # Criar DataLoaders para treino e validação
            train_dataset = TensorDataset(
                train_encodings["input_ids"],
                train_encodings["attention_mask"],
                torch.tensor(y_train)
            )
            val_dataset = TensorDataset(
                val_encodings["input_ids"],
                val_encodings["attention_mask"],
                torch.tensor(y_val)
            )
            
            train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
            val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

            # Definir o otimizador
            optimizer = AdamW(model.parameters(), lr=1e-5)
            model.to(self.device)

            print(f"  - iniciando o treinamento no fold {fold + 1}")
            
            # Loop de treinamento por epoch
            for epoch in range(epochs):
                model.train()
                total_train_loss = 0  # Para acumular a perda do treinamento
                l = len(train_dataloader)

                for i, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, labels = batch
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_train_loss += loss.item()  # Soma a perda de cada batch
                    loss.backward()
                    optimizer.step()

                    # Exibir progresso
                    print(f"  - epoch {epoch + 1}/{epochs}, batch {i + 1}/{l}, loss: {loss.item():.12f}", end="\r")

                # Cálculo da perda média de treinamento
                avg_train_loss = total_train_loss / l
                print(f"\n  - Epoch {epoch + 1}/{epochs} - Loss médio no treinamento: {avg_train_loss:.12f}")

                # Avaliação no conjunto de validação
                model.eval()
                total_val_loss = 0  # Para acumular a perda de validação
                predictions = []
                true_labels = []

                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, labels = batch

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        logits = outputs.logits
                        total_val_loss += loss.item()

                        # Armazenar predições e labels verdadeiros
                        probs = torch.softmax(logits, dim=-1)
                        predictions.extend(torch.argmax(probs, dim=-1).tolist())
                        true_labels.extend(labels.tolist())

                # Cálculo da perda média de validação
                avg_val_loss = total_val_loss / len(val_dataloader)
                
                # Comparação para evitar overfitting
                if avg_val_loss < best_validation_loss and avg_train_loss < avg_val_loss:
                    best_validation_loss = avg_val_loss
                    self.model = model
                    print(f"  - Novo melhor modelo salvo! Validation Loss: {best_validation_loss:.12f}, Train Loss: {avg_train_loss:.12f}")
                    if best_validation_loss <= 0.001:
                        break
                else:
                    print(f"  - Modelo não substituído. Test Loss: {avg_val_loss:.12f}, Train Loss: {avg_train_loss:.12f}")

                # Relatório de classificação
                print(f"  - Loss médio no teste melhor modelo: {best_validation_loss:.12f}")
                print(classification_report(true_labels, predictions))

        # Após todos os folds, salvar o modelo se solicitado
        if save_model:
            torch.save(model, model_dir)
            print(f"Modelo salvo como {model_dir}")
    
        print(f"\nTreinamento por CV {k_folds} folds completo.")

    def train(
        self,
        X: pd.Series,
        y: pd.Series,
        epochs: int = 5,
        batch_size: int = 8,
        test_size: float = 0.2,
        random_state: int = 42,
        save_model: bool = True,
        model_dir: str = "bert_model.pth",
    ):
        """
        (PT-BR)
        Descrição:
            Função que executa o treinamento do modelo BERT.
        Args:
            X (pandas.core.series.Series): Series do pandas contendo os textos
            y (pandas.core.series.Series): Series do pandas contendo  as labels no formato integer (0, 1, ...)
            epochs (int): Número de epochs utilizados no treinamento. Defaults to 5.
            batch_size (int): Tamanho do lote utilizado no treinamento. Defaults to 8.
            test_size (float): Proporção utilizado para teste para validação pós treinamento. Defaults to 0.2.
            random_state (int): Semente do gerador de números aleatórios utilizado para separação dos conjuntos de treinamento e teste. Defaults to 42.
            save_model (bool): Condição para salvar modelo. Defaults to True.
            model_dir (str): Nome do modelo salvo. Defaults to "bert_model.pth".
        Retorna:
            model (BertForSequenceClassification): Modelo BERT treinado.

        (EN-US)
        Description:
            Function that performs the training of the BERT model.
        Args:
            X (pandas.core.series.Series): Pandas Series containing the texts
            y (pandas.core.series.Series): Pandas Series containing the labels in integer format (0, 1, ...)
            epochs (int): Number of epochs used in training. Defaults to 5.
            batch_size (int): Batch size used in training. Defaults to 8.
            test_size (float): Proportion used for testing for post-training validation. Defaults to 0.2.
            random_state (int): Random seed used for splitting training and testing sets. Defaults to 42.
            save_model (bool): Condition to save the model. Defaults to True.
            model_dir (str): Name of the saved model. Defaults to "bert_model.pth".
        Returns:
            model (BertForSequenceClassification): Trained BERT model.
        """
        print("Treinamento do modelo iniciado")
        # Separar conjunto de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(
            f"  - train_test_split com {X_train.shape[0]} para treino e {X_test.shape[0]} para teste"
        )

        # Carregar o modelo BERT pré-treinado e o tokenizador
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )  # 2 classes: 0 and 1

        # Tokenizar os textos
        train_encodings = self.tokenizer(
            list(X_train),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        print(f"  - tokenização do conjunto de treinamento")
        test_encodings = self.tokenizer(
            list(X_test),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        print(f"  - tokenização do conjunto de teste")
        # Criar DataLoader para o treinamento
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            torch.tensor(y_train.values),
        )
        print(f"  - criação do dataset do conjunto de treinamento")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size
        )

        # Criar DataLoader para o teste
        test_dataset = TensorDataset(
            test_encodings["input_ids"],
            test_encodings["attention_mask"],
            torch.tensor(y_test.values),
        )
        print(f"  - criação do dataset do conjunto de teste")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=batch_size
        )
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model.to(self.device)
        print("")
        print("  treinando...")

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            l = len(train_dataloader)
            for i, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                print_progress_bar(i + 1, l, prefix=f"  - epoch {epoch + 1}: ")
            print(f"    average training loss: {total_loss / len(train_dataloader)}")
            
            total_loss = 0
            for i, batch in enumerate(test_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Average test loss: {total_loss / len(test_dataloader)}")

        if save_model:
            torch.save(model, model_dir)
            print(f"Modelo salvo como {model_dir}")
        
        model.eval()
        predictions = []
        true_labels = []
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels = batch
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predictions.extend(torch.argmax(probs, dim=-1).tolist())
                true_labels.extend(labels.tolist())
        
        results = []
        print(classification_report(true_labels, predictions))
        self.model = model

    def load_model(self, path_to_model: str):
        """
        (PT-BR)
        Descrição:
            Carrega o modelo salvo no treinamento.
        Args:
            path_to_model (str): Caminho até o modelo salvo em .pth.
        Retorna:
            None

        (EN-US)
        Description:
            Loads the saved model from training.
        Args:
            path_to_model (str): Path to the saved model in .pth format.
        Returns:
            None
        """
        self.model = torch.load(path_to_model, map_location=self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        print(f"Modelo carregado com sucesso!")

    def df_infer(
        self,
        df: pd.DataFrame,  # Dataframe do pandas contendo pelo menos uma coluna com os textos que serão classificados,
        X_col_nm: str = "X",  # nome da coluna que contém os textos
        batch_size: int = 10,
        y_infer_col_nm: str = "y_infered",
        prob_column_name: str = "prob",
    ) -> (
        pd.DataFrame
    ):  # DataFrame do pandas contendo uma coluna com o texto (tendo o mesmo nome do input original), 'y_infered' com a previsão e 'prob' com a probabilidade da classe positiva
        """
        (PT-BR)
        Descrição:
            Função que executa a inferência a partir do DataFrame do pandas fornecido.

        Args:
            df (pandas.core.frame.DataFrame): Dataframe do pandas contendo pelo menos uma coluna com os textos que serão classificados
            X_col_nm (str): Nome da coluna que contém os textos que serão classificados pelo BERT. Defaults to 'X'
            batch_size (int): Tamanho do lote utilizado na inferência. Defaults to 8

        Returns:
            pandas.core.frame.DataFrame: DataFrame do pandas contendo uma coluna 'X' com o texto, 'y_infered' com a previsão e 'prob' com a probabilidade da classe positiva
        
        (EN-US)
        Description:
            Function that performs inference from the provided pandas DataFrame.

        Args:
            df (pandas.core.frame.DataFrame): Pandas DataFrame containing at least one column with the texts to be classified
            X_col_nm (str): Name of the column containing the texts to be classified by BERT. Defaults to 'X'
            batch_size (int): Batch size used in inference. Defaults to 8

        Returns:
            pandas.core.frame.DataFrame: Pandas DataFrame containing a column 'X' with the text, 'y_infered' with the prediction, and 'prob' with the probability of the positive class
        """
        if not self.tokenizer or not self.model:
            print("(PT-BR) Tokenizer ou o modelo não foram definidos. Treine o modelo utilizando o método train(X, y) antes de aplicar a inferência!")
            print("(EN-US) Tokenizer or model not defined. Train the model using the train(X, y) method before applying inference!")
            return None

        X = df[X_col_nm]
        val_encodings = self.tokenizer(
            list(X), padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inference_dataset = TensorDataset(
            val_encodings["input_ids"], val_encodings["attention_mask"]
        )
        inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size)
        probs_list = []
        preds_list = []

        for batch in inference_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask = batch

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                probs_list += probs[:, 1].tolist()
                preds_list += torch.argmax(probs, dim=-1).tolist()

        df[y_infer_col_nm] = preds_list
        df[prob_column_name] = probs_list
        return df

def main():
    model = BERTHuggingFaceClassification()
    model.load_model("bert_model.pth")
    output = model.infer(
        pd.DataFrame(
            {
                "X": [
                    "Melhor que nem tivessem feito esse filme.",
                    "Péssima adaptação.",
                    "Uma obra prima do cinema.",
                    "O diretor perdeu a noção de liderança com o elenco. Não sabiam os próprios papéis...",
                    "Irei assistir em looping.",
                    "Ótimo pra quem não tem amor à vida.",
                ]
            }
        )
    )
    print(output)  
    return None

if __name__ == "__main__":
    main()
"""
Funções utilitárias para acesso aos recursos Google
"""


# Google Authentication
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from pydrive.files import GoogleDriveFile
import base64
from google.auth import exceptions
from google.oauth2 import service_account
#from oauth2client import service_account

# Sys miscellaneous
import os 
import json

# Data Wrangling
import pandas as pd
import numpy as np
import math 
from pandas import util
import re
import datetime as dt

def show_drive_list(file_list):
    """
    
    """
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))
    return None

def get_drive_list(drive, file_id = "root"):
    """
    
    """
    file_list = drive.ListFile({'q': f"'{file_id}' in parents and trashed=false"}).GetList()

    list_drive = []
    for i in range(len(file_list)):
        list_drive.append(file_list[i]['title'])
    list_drive.sort()
    return list_drive, file_list

def cloud_items_check(list_dir, drive_list):
    """
    
    """
    list_dir = os.listdir(list_dir)
    itens_distintos = list(set(drive_list) - set(list_dir))
    itens_distintos.sort()
    
    return itens_distintos

def download_items(list_dir, drive_list, file_list):
    """
    
    """
    if not os.path.exists(list_dir):
        os.mkdir(list_dir)

    lst_distinct = cloud_items_check(list_dir = list_dir, 
                                     drive_list = drive_list)

    if len(lst_distinct) > 0:
        for i in range(len(lst_distinct)):
            print("\nBuscando arquivo '{}'".format(lst_distinct[i]))
            
            for file in file_list:
                if file['title'] == lst_distinct[i]:
                    if file['mimeType'].startswith('application/vnd.google-apps'):
                        print("Ignorando arquivo inválido de download por ser do tipo '{}'".format(file['mimeType']))
                        break
                    else:
                        print("Iniciando download do arquivo '{}'".format(lst_distinct[i]))
                        filename = os.path.join(list_dir, file['title'])
                        file.GetContentFile(filename)
                        print("{0} - {1}/{2}".format(file['title'], i+1, len(lst_distinct)))
                        break
                else:
                    continue
    else:
        print('Todos os itens de mesmo nome já existem no diretório: ')
        print(f"'{list_dir}'")
        print('Prosseguindo com a análise...')
    return None
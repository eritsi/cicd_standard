# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ! pip install boxsdk

import sys
sys.path.append('./utils/')
import box

folder_id = '150950398192'
AccessToken = '' 
client_id = ''
client_secret = ''

client = box.GetClientObject(AccessToken, client_id, client_secret)

box.read_folder(client, folder_id)

file_id = '889172215319'

box.read_file(client, file_id, 1, [0, 1])

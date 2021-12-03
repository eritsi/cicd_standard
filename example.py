# -*- coding: utf-8 -*-
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

project_id = ''
dataset_id = ''
AccessToken = '' 
client_id = ''
client_secret = ''

client = box.GetClientObject(AccessToken, client_id, client_secret)

box.list_folder(client, '150950398192') # 2014年度

eng_cols = ['client_year', 'month', 'company_code', 'plant_code', 'space_code', 'tana_code', 'MTyp', 'account_code',
           'layer_code', 'product', 'product_name', 'WBS', 'supplier_code', 'cost', 'stock_count', 'unit', 'stock_price', 
           'ABC', 'purchase_code', 'cost_next_term', 'stock_price_next_term', 'company_name', 'plant', 'storage',
           'Mtyp_name', 'product_type', 'supplier_name', 'purchase_name', 'kW', 'category', 'category_name']


def read_folder_2014(_client, _folder_id, _project_id, _dataset_id, eng_cols):
    items = _client.folder(_folder_id).get_items()
    _file_ids = []
    for item in items:
        _file_ids.append(item.id)
        data = box.read_file(_client, item.id, 4, [0])
        table_name = _dataset_id + ".zaiko_" + box.get_filename(_client, item.id)[0:6] + "01"
        data[0].iloc[:, 1:].set_axis(eng_cols, axis=1).to_gbq(destination_table=table_name, project_id=_project_id, if_exists='replace')
        print('{0} {1} is loaded to BQ as "{2}"'.format(item.type.capitalize(), item.id, table_name))


read_folder_2014(client, '150950398192', project_id, dataset_id, eng_cols) # 2014年度は6月からフォーマットが変わるので対処が必要


def read_folder_2015(_client, _folder_id, _project_id, _dataset_id, eng_cols):
    items = _client.folder(_folder_id).get_items()
    _file_ids = []
    for item in items:
        _file_ids.append(item.id)
        data = box.read_file(_client, item.id, 1, [0])
        table_name = _dataset_id + ".zaiko_" + box.get_filename(_client, item.id)[0:6] + "01"
        data[0].iloc[:, :].set_axis(eng_cols, axis=1).to_gbq(destination_table=table_name, project_id=_project_id, if_exists='replace')
        print('{0} {1} is loaded to BQ as "{2}"'.format(item.type.capitalize(), item.id, table_name))


read_folder_2015(client, '150949643623', project_id, dataset_id, eng_cols) # 2015年度は新フォーマットで全月ok

read_folder_2015(client, '150950501994', project_id, dataset_id, eng_cols) # 2016年度は2015年度と同じフォーマットで全月ok

read_folder_2015(client, '150949946822', project_id, dataset_id, eng_cols) # 2017年度は2015年度と同じフォーマットで全月ok

read_folder_2014(client, '149757376225', project_id, dataset_id, eng_cols) # 2018年度は2014年度と同じフォーマットで全月ok

read_folder_2015(client, '151117266320', project_id, dataset_id, eng_cols) # 2019年度は2015年度と同じフォーマットで全月ok

read_folder_2015(client, '149894337466', project_id, dataset_id, eng_cols) # 2020年度は2015年度と同じフォーマットで全月ok

read_folder_2015(client, '151358991850', project_id, dataset_id, eng_cols) # 2021年度は2015年度と同じフォーマットで全月ok


def read_folder_201406(_client, _folder_id, _project_id, _dataset_id, eng_cols):
    items = _client.folder(_folder_id).get_items()
    _file_ids = []
    for item in items:
        _file_ids.append(item.id)
        if item.id =='889172917124' or item.id =='889159434370':
            print('{0} {1} is skipped...'.format(item.type.capitalize(), item.id))
        else:
            data = box.read_file(_client, item.id, 1, [0])
            table_name = _dataset_id + ".zaiko_" + box.get_filename(_client, item.id)[0:6] + "01"
            data[0].iloc[:, :].set_axis(eng_cols, axis=1).to_gbq(destination_table=table_name, project_id=_project_id, if_exists='replace')
            print('{0} {1} is loaded to BQ as "{2}"'.format(item.type.capitalize(), item.id, table_name))


read_folder_201406(client, '150950398192', project_id, dataset_id, eng_cols) # 2014年度は6月からフォーマットが変わるので対処が必要

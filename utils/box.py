from boxsdk import OAuth2, Client
import pandas as pd

def GetClientObject(_AccessToken, _client_id, _client_secret):
    
    # ユーザー認証
    oauth = OAuth2(
        client_id = _client_id,
        client_secret = _client_secret,
        store_tokens = None,
        access_token =_AccessToken
    )
    # 権限情報をインスタンス化
    client = Client(oauth)
    print('BoxAppへの認証完了')
    
    return client

def read_folder(_client, _folder_id):
    items = _client.folder(_folder_id).get_items()
    _file_ids = []
    for item in items:
        _file_ids.append(item.id)
        print('{0} {1} is named "{2}"'.format(item.type.capitalize(), item.id, item.name))

def read_file(_client, _file_id, _header, _sheet_name):
    
    download_url = _client.file(_file_id).get_download_url()
    _excel_data = pd.read_excel(download_url, _header, _sheet_name)
    
    return _excel_data
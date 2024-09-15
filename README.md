# cicd_standard
起点リポジトリ。 これをimport repositoryして使い捨てる
具体的には
```
git remote -v
rm -rf .git
git init
git remote add origin {New Repository}
```
マージ済みのコミット取り消しコマンドもついでに備忘録  
```
git reset --hard <戻りたいコミットのハッシュ>
git push -f origin main
```  
この時、間のコミットで変化したファイルがGitPod上でもリセットされるので、実行前に維持したい変更部位をクリップボードに入れておくこと  

コミット・タグを指定してその時点でのツリーだけをgit cloneしてくる  
[参考](https://yo.eki.do/notes/git-only-single-commit/)
```
mkdir foo
cd foo
git init
git remote add origin <cloneする元のURL>
git fetch --depth 1 origin <コミットを指定するSHA1ハッシュやタグ名>
git reset --hard FETCH_HEAD
```

## Activate github actions in the new repository
- Settings -> Actions -> Allow all actions

## Open repository with gitpod
- google-cloud-sdk のインストール
- gcloud コマンドの初期設定
- QAに答えつつ、認証urlをオープンして適切なgoogleアカウントを選択し、トークンをgitpodへコピーする

## Action will be triggered.
.sql や .py ファイルを作成し、pushするとフォーマッターやリンターが動く。

### SQL formatter : GitHub Action
     zetasql-formatter.ymlを設置する方式にした。
     以下を使う手もある
     https://github.com/yoheikikuta/sql-autoformat-action
     
### autopep8 linter : GitHub Action
     https://github.com/marketplace/actions/autopep8#automated-pull-requests


### GCS系のコードメモ。そのうちutilsに入れる
```
! pip install google-cloud-storage

from google.cloud import storage
from google.oauth2 import service_account
import os
import json
from pprint import pprint

# key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test-gcs.json')
# service_account_info = json.load(open(key_path))
# credentials = service_account.Credentials.from_service_account_info(service_account_info)
client = storage.Client(
    project='inz-dash-dev'
)

# get list
buckets = client.list_buckets()
for obj in buckets:
    print('-------->')
    pprint(vars(obj))
    # get
    bucket = client.get_bucket(obj.id)
    print('\t-------->')
    pprint(vars(bucket))

import pandas as pd
from io import BytesIO

bucket_name = 'inz-dash-bucket'
blob_name = 'picking_raw_data_predict/20220915/0900-upload-estimation-csv-062745.csv'
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)

content = blob.download_as_string()

# print("read: [{}]".format(content.decode('utf-8')))
test = pd.read_csv(BytesIO(content))
test.head()

import chardet


filepath = "集品【生産性分析用】0916-0923_1800.csv"
with open(filepath, 'rb') as f:
    c = f.read()
    result = chardet.detect(c)
```

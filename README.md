# cicd_standard
起点リポジトリ。 これをimport repositoryして使い捨てる
具体的には
```
git remote -v
rm -rf .git
git init
git remote add origin {New Repository}
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

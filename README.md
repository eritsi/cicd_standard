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

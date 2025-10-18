
# visualize_map
RSデータセットを使った土地被覆分類タスクにおける各プロセスでの実行結果を可視化する。
<!--
```
visualize_train_test_split_map() - RSデータセット(X,y)のtrain, testの空間的配置の可視化
visualize_iteration_map()        - Self-Trainingでの各反復におけるラベル付きデータの可視化
visualize_prediction_map()       - modelの分類結果の可視化
```
-->
|関数                            |説明                                                 |
|--------------------------------|-----------------------------------------------------|
|visualize_train_test_split_map() | RSデータセット(X,y)のtrain, testの空間的配置の可視化  |
|visualize_iteration_map()        | Self-Trainingでの各反復におけるラベル付きデータの可視化|
|visualize_prediction_map()       | modelの分類結果の可視化                              |

## 概要(Overview)

## インストール方法(Installation)
```bash
pip install git+https://github.com/Picminmin/visualize_map
```
インストールしたパッケージをGitHubリポジトリの最新版に置き換えたい場合には、以下のコマンドを実行してください。
```
pip install -U git+https://github.com/Picminmin/visualize_map
```

## 使い方(Usage / Examples)
```python
# Python のコード例
from visualize_map.visualize import (
visualize_train_test_split_map,
visualize_iteration_map,
visualize_prediction_map
)
```
```python
# 例: train/test split の可視化
visualize_train_test_split_map(
y, train_mask, test_mask,
dataset_keyword="Indianpines",
save_dir="img"
)
```


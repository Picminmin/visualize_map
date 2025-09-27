
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# visualize_map.py の場所を基準にする
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RS_CSV_ROOT = os.path.join(os.path.dirname(THIS_DIR), "rs_csv")

def visualize_train_test_split_map(
    y,
    train_mask,
    test_mask,
    background_label=0,
    dataset_keyword=None,
    save_dir="img",
    title="Train/Test Spatial Split Visualization",
    rs_csv_root=RS_CSV_ROOT
):
    """
    教師データとテストデータの空間的配置を可視化し、棒グラフでクラスごとの分布を表示する。
    dataset_keyword に対応する rs_csv サブディレクトリ内の CSV ファイルをカラーマップ定義に使用。

    from RS_GroundTruth.rs_dataset import RemoteSensingDataset  # あなたのrs_dataset.py
    rs = RemoteSensingDataset()  ・・・①

    Args:
        y (ndarray): ラベルマップ (H, W)
        train_mask (ndarray): 教師データのマスク (H, W)
        test_mask (ndarray): テストデータのマスク (H, W)
        background_label (int): 背景クラスラベル
        dataset_keyword (str): rs_csv 内のサブディレクトリ名
        save_dir (str): 結果画像の保存先
        title (str): 図のタイトル
        rs_csv_root (str): rs_csv ディレクトリのルート
    """
    os.makedirs(save_dir, exist_ok=True)

    unique_classes = np.unique(y[y != background_label])
    n_classes = len(unique_classes)

    # ==============================
    # (1) カラーマップを読み込み
    # ==============================
    color_dict, name_dict = load_colormap_from_csv(dataset_keyword=dataset_keyword)

    if color_dict:
        base_colors = [to_rgba(color_dict.get(cls, "#808080")) for cls in unique_classes]
        labels = [name_dict.get(cls, f"Class {cls}") for cls in unique_classes]
    else:
        print("[WARN] CSVが利用できないため、デフォルトのcmapを使用します。")
        cmap = plt.cm.get_cmap("tab20", n_classes)
        base_colors = [cmap(i) for i in range(n_classes)]
        labels = [f"Class {cls}" for cls in unique_classes]

    # ==============================
    # (2) 空間オーバーレイ図
    # ==============================
    display = np.full((*y.shape, 4), np.nan)
    for idx, cls in enumerate(unique_classes):
        cls_mask = (y == cls)
        display[cls_mask] = to_rgba(base_colors[idx], alpha=0.6)
        display[np.logical_and(cls_mask, train_mask)] = to_rgba(base_colors[idx], alpha=1.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(f"{title}\nTrain={np.sum(train_mask)}, Test={np.sum(test_mask)}", fontsize=12)
    plt.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=base_colors[i], markersize=10)
        for i in range(n_classes)
    ]
    plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)

    overlay_path = os.path.join(save_dir, f"{dataset_keyword}_spatial_split_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ==============================
    # (3) クラスごとのサンプル数棒グラフ
    # ==============================
    train_counts = [np.sum(np.logical_and(y == cls, train_mask)) for cls in unique_classes]
    test_counts = [np.sum(np.logical_and(y == cls, test_mask)) for cls in unique_classes]

    x = np.arange(n_classes)
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, train_counts, bar_width, label="Train", color="#1f77b4")
    plt.bar(x + bar_width / 2, test_counts, bar_width, label="Test", color="#d62728")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Sample Count")
    plt.title(f"{dataset_keyword}:Class-wise Train/Test Distribution")
    plt.legend()

    # 棒グラフ上にサンプル数を表示
    for i, v in enumerate(train_counts):
        plt.text(i - bar_width/2, v + 1, str(v), ha="center", fontsize=9)
    for i, v in enumerate(test_counts):
        plt.text(i + bar_width/2, v + 1, str(v), ha="center", fontsize=9)

    bargraph_path = os.path.join(save_dir, f"{dataset_keyword}_class_distribution.png")
    plt.tight_layout()
    plt.savefig(bargraph_path, dpi=300)
    plt.close()
    print(f"[INFO] {dataset_keyword}の教師データ・テストデータの空間的な配置の可視化")
    print(f"[INFO] オーバーレイ図を保存しました: {overlay_path}")
    print(f"[INFO] クラス分布棒グラフを保存しました: {bargraph_path}")

def visualize_iteration_map(y, expand_label, L_index, U_index,
                            image_shape, save_path, dataset_keyword=None,
                            rs_csv_root=RS_CSV_ROOT, title="Iteration Map",
                            background_label=0):
    """
    ある反復におけるラベル付きデータのオーバーレイ図を保存。

    from RS_GroundTruth.rs_dataset import RemoteSensingDataset  # あなたのrs_dataset.py
    rs = RemoteSensingDataset()  ・・・①


    Args:
        y (ndarray): ground truth (flattened)
        expand_label (ndarray): 現在のラベル配列 (flattened)
        L_index (ndarray): ラベル付きデータインデックス
        U_index (ndarray): ラベルなしデータインデックス
        image_shape (tuple): (H, W)
        save_path (str): 保存先ファイル名
        dataset_keyword (str): ①について、rs.available_data_keywordから参照可。
        title (str, optional): _description_. Defaults to "Iteration Map".
        background_label (int, optional): 背景ラベル
    """
    # 保存先ディレクトリを自動作成(ディレクトリパスに対してmakedirsを呼ぶ)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W = image_shape
    unique_classes = np.unique(y[y != background_label])

    # --- カラーマップ読み込み ---
    try:
        if dataset_keyword is not None:
            color_dict, name_dict = load_colormap_from_csv(dataset_keyword=dataset_keyword, rs_csv_root=rs_csv_root)
            base_colors = {cls: to_rgba(color_dict.get(cls, "#808080")) for cls in unique_classes}
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f"[WARN] CSV読み込み失敗 → デフォルトcmap使用 ({e})")
        cmap = plt.cm.get_cmap("tab20", len(unique_classes))
        base_colors = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    # ディスプレイ配列
    display = np.ones((H, W, 4)) * 1.0 # 白背景
    for cls in unique_classes:
        cls_mask = (expand_label.reshape(H, W) == cls)
        display[cls_mask] = base_colors[cls]

    # U_index のマスクを作成
    mask_U = np.zeros(H * W, dtype=bool)
    mask_U[U_index] = True
    mask_U = mask_U.reshape(H, W)
    # 未ラベル部分をグレーに塗る
    display[mask_U] = (0.8, 0.8, 0.8, 1.0)

    plt.figure(figsize=(8,8))
    plt.imshow(display)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def visualize_prediction_map(y_pred, test_index, image_shape, save_path,
                             dataset_keyword=None,rs_csv_root=RS_CSV_ROOT,
                             background_label=0):
    """
    テストデータ領域だけの最終予測結果を可視化する。

    Args:
        y_pred (ndarray): shape = (n_test,) の予測ラベル
        test_index (ndarray): テストデータのフラットインデックス
        image_shape (tuple): (H, W)
        save_path (str): 保存先パス
    """
    # 保存先ディレクトリを自動作成(ディレクトリパスに対してmakedirsを呼ぶ)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = image_shape
    # 全体を背景ラベルで初期化
    pred_full = np.full(H * W, background_label, dtype=int)
    # テストデータ領域だけ予測ラベルを代入
    pred_full[test_index] = y_pred
    pred_2d = pred_full.reshape(H, W)

    # 可視化対象クラス (背景を除く)
    unique_classes = np.unique(pred_2d[pred_2d != background_label])

    # --- カラーマップ読み込み ---
    try:
        if dataset_keyword is not None:
            color_dict, name_dict = load_colormap_from_csv(dataset_keyword=dataset_keyword, rs_csv_root=rs_csv_root)
            base_colors = {cls: to_rgba(color_dict.get(cls, "#808080")) for cls in unique_classes}
        else:
            raise FileNotFoundError
    except Exception as e:
        print(f"[WARN] CSV読み込み失敗 → デフォルトcmap使用 ({e})")
        cmap = plt.cm.get_cmap("tab20", len(unique_classes))
        base_colors = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    display = np.ones((H, W, 4)) * 1.0
    for cls in unique_classes:
        cls_mask = (pred_2d == cls)
        display[cls_mask] = base_colors[cls]

    plt.figure(figsize=(8,8))
    plt.imshow(display)
    plt.title(f"Final Prediction Map(Test Only):{dataset_keyword}")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print("visualize_prediction_map完了。")

def load_colormap_from_csv(dataset_keyword, rs_csv_root=RS_CSV_ROOT):
    """
    カラーマップ読み込みの共通関数
    rs_csv/<dataset_keyword>/ 内にある CSV を読み込み、
    class_id + color, class_id → name の辞書を返す。

    Returns:
        color_dict (dict), name_dict (dict)
    """
    csv_dir = os.path.join(rs_csv_root, dataset_keyword)
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"[ERROR] ディレクトリが存在しません: {csv_dir}")

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if len(csv_files) != 1:
        raise RuntimeError(f"[ERROR] {csv_dir} には CSV ファイルが１つだけ存在する必要があります。見つかった数: {len(csv_files)}")

    csv_path = os.path.join(csv_dir, csv_files[0])
    df = pd.read_csv(csv_path)

    color_dict = dict(zip(df["class_id"], df["color"]))
    name_dict = dict(zip(df["class_id"], df["name"]))

    return color_dict, name_dict

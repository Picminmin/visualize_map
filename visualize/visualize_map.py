
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def visualize_map(
    y,
    train_mask,
    test_mask,
    background_label=0,
    csv_path=None,
    save_dir="img",
    title="Train/Test Spatial Split Visualization"
):
    """
    教師データ・テストデータの空間オーバーレイ図とクラス分布棒グラフを作成する。
    CSVファイルでカラーマップを定義可能。

    Args:
        y (ndarray): ラベルマップ (H, W)
        train_mask (ndarray): 教師データのマスク (H, W)
        test_mask (ndarray): テストデータのマスク (H, W)
        cfg.background_label (int): 背景クラスラベル
        csv_path (str): カラーマップ定義CSVのパス
        save_dir (str): 結果画像の保存先
        title (str): 図のタイトル
    """
    os.makedirs(save_dir, exist_ok=True)

    unique_classes = np.unique(y[y != background_label])
    n_classes = len(unique_classes)

    # ==============================
    # (1) カラーマップを読み込み
    # ==============================
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        color_dict = dict(zip(df["class_id"], df["color"]))
        name_dict = dict(zip(df["class_id"], df["name"]))
        base_colors = [to_rgba(color_dict.get(cls, "#808080")) for cls in unique_classes]
        labels = [name_dict.get(cls, f"Class {cls}") for cls in unique_classes]
    else:
        print("[WARN] CSVファイルが見つからないため、デフォルトのcmapを使用します。")
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

    overlay_path = os.path.join(save_dir, "spatial_split_overlay.png")
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
    plt.title("Class-wise Train/Test Distribution")
    plt.legend()

    # 棒グラフ上にサンプル数を表示
    for i, v in enumerate(train_counts):
        plt.text(i - bar_width/2, v + 1, str(v), ha="center", fontsize=9)
    for i, v in enumerate(test_counts):
        plt.text(i + bar_width/2, v + 1, str(v), ha="center", fontsize=9)

    bargraph_path = os.path.join(save_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(bargraph_path, dpi=300)
    plt.close()

    print(f"[INFO] オーバーレイ図を保存しました: {overlay_path}")
    print(f"[INFO] クラス分布棒グラフを保存しました: {bargraph_path}")

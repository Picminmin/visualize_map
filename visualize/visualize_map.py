
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def visualize_iteration_map(
    y,
    expand_label,
    L_index,
    U_index,
    image_shape,
    save_path,
    dataset_keyword=None,
    rs_csv_root=RS_CSV_ROOT,
    title="Iteration Map",
    background_label=0,
    boundary = None,
    boundary_color = (1.0, 0.3, 0.3, 0.4) # RGBA 薄赤 (透過度0.4)
):
    """
    ある反復におけるラベル付きデータ・ラベルなしデータ・(オプションで)
    エッジ領域を可視化する。

    Args:
        y (ndarray): 正解ラベル(flatten)
        expand_label (ndarray): ラベル拡張後のラベル (flatten)
        L_index (ndarray): ラベル付きデータインデックス
        U_index (ndarray): ラベルなしデータインデックス
        image_shape (tuple): (H, W)
        save_path (str): 画像保存先パス
        dataset_keyword (str): カラーマップ指定用データセット名
        rs_csv_root (str, optional): カラーマップCSVルート
        title (str, optional): 図タイトル
        background_label (int, optional): 背景ラベル値
        boundary (ndarray, optional): エッジ領域 (H, W) {0,1}
        boundary_color (tuple, optional): RGBA形式の重畳色
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

    # --- 基本表示 ---
    display = np.ones((H, W, 4)) # 白背景
    for cls in unique_classes:
        cls_mask = (expand_label.reshape(H, W) == cls)
        display[cls_mask] = base_colors[cls]

    # --- 未ラベルをグレー塗り ---
    mask_U = np.zeros(H * W, dtype=bool)
    mask_U[U_index] = True
    mask_U = mask_U.reshape(H, W)
    display[mask_U] = (0.8, 0.8, 0.8, 1.0)
    # --- boundary overlay (オプション) ---
    if boundary is not None:
        if boundary.shape != (H, W):
            raise ValueError(f"boundary shape mismatch: expected {(H, W)}, got {boundary.shape}")
        boundary_mask = boundary.astype(bool)

        # display[..., c]: 元の画像(RGBA配列)のチャンネルのcの値
        # boundary_color[c]: 指定した赤色の各チャンネル値 ((1.0, 0.3, 0.3, 0.4))
        # boundary_color[3]: RGBAのα値 (透過度) = たとえば0.4
        for c in range(3):
            # 各チャンネル(0:R, 1:G, 2:B)について順に半透明ブレンド処理を行う
            display[..., c] = np.where(boundary_mask,
                                       boundary_color[c]*boundary_color[3] + display[..., c] * (1 - boundary_color[3]),
                                       display[..., c])
        display[..., 3] = np.where(boundary_mask,
                                   np.maximum(display[..., 3], boundary_color[3]),
                                    display[..., 3])
        # Q. np.where(condition, x, y)とは何か。
        # A.condition[i,j]がTrueのときはx[i,j]を、
        # Falseのときはy[i,j]を結果に採用する。

    # --- 描画 ---
    plt.figure(figsize=(8,8))
    plt.imshow(display)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Visualization saved → {save_path}")

def visualize_prediction_map(
    y_pred,
    pred_index,
    image_shape,
    save_path,
    dataset_keyword=None,
    y_true=None,
    rs_csv_root=RS_CSV_ROOT,
    background_label=0,
    save_confusion=False,
    # overlay=False,
    # X_rgb=None, # overlay=Trueの場合に元画像(RGB)を渡す
):
    """
    教師データまたはテストデータに対する予測結果を可視化する。
    - y_true が与えられた場合 : 教師データ上での正解(青)/誤分類(赤)/背景(白)を可視化。
    - y_true が None の場合 : 従来通りの予測結果カラーマップを可視化。
    - save_confusion=True: 混同行列を画像として保存
    - overlay=True: 元画像(X_rgb)を結果を重ね合わせ表示

    Args:
        y_pred (ndarray): shape = (n_pred,) の予測ラベル
        pred_index (ndarray): 予測対象画素のフラットインデックス
        image_shape (tuple): (H, W)
        save_path (str): 保存先パス
        dataset_keyword (str, optional): データセット名
        y_true (ndarray, optional): 教師データ上の真のラベル
        rs_csv_root (str, optional): カラーマップCSVのルートパス
        background_label (int, optional): 背景ラベル
        save_confusion (bool): 混同行列を保存するか
        overlay (bool): 元画像と重ね合わせを行うか
        X_rgb (ndarray): (H, W, 3) のRS画像 (overlay=Trueの場合必須)
    """

    # 保存先ディレクトリを自動作成(ディレクトリパスに対してmakedirsを呼ぶ)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W = image_shape

    # --- 1. ベース画像を背景ラベルで初期化 ---
    pred_full = np.full(H * W, background_label, dtype=int)
    pred_full[pred_index] = y_pred # predした領域の予測ラベルを代入
    pred_2d = pred_full.reshape(H, W)

    if y_true is not None:

        # ============================================================
        # 教師データ上の可視化:正解(青)/誤分類(赤)
        # ============================================================
        true_full = np.full(H * W, background_label, dtype = int)
        true_full[pred_index] = y_true
        true_2d = true_full.reshape(H, W)

        correct_mask = (pred_2d == true_2d) & (true_2d != background_label)
        wrong_mask = (pred_2d != true_2d) & (true_2d != background_label)
        # --- 誤分類率を算出 ---
        n_correct = np.count_nonzero(correct_mask)
        n_wrong = np.count_nonzero(wrong_mask)
        n_total = n_correct + n_wrong
        error_rate = n_wrong / n_total if n_total > 0 else 0.0
        acc_rate = 1 - error_rate

        # --- RGBオーバーレイまたは純粋マスク ---
        # if overlay and X_rgb is not None:
            # base_img = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min() + 1e-8)
            # overlay_img = base_img.copy()
            # overlay_img[correct_mask] = 0.5 * base_img[correct_mask] + 0.5 * np.array([0.2, 0.4, 1.0])
            # overlay_img[wrong_mask] = 0.5 * base_img[wrong_mask] + 0.5 * np.array([1.0, 0.2, 0.2])
            # display = overlay_img
        # else:
            # display = np.ones((H, W, 3)) # 白背景
            # display[correct_mask] = [0.2, 0.4, 1.0] # 青: 正解
            # display[wrong_mask] = [1.0, 0.2, 0.2]   # 赤: 誤分類

        display = np.ones((H, W, 3)) # 白背景
        display[correct_mask] = [0.2, 0.4, 1.0] # 青: 正解
        display[wrong_mask] = [1.0, 0.2, 0.2]   # 赤: 誤分類
        plt.figure(figsize=(8,8))
        plt.imshow(display)
        plt.title(
            f"Train Prediction Map (Correct/Incorrect)\n"
            f"Acc: {acc_rate*100:.2f}%, Error: {error_rate*100:.2f}% ({dataset_keyword})"
        )
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", dpi = 300)
        plt.close()
        print(f"[INFO] 教師データ可視化を保存しました → {save_path}")
        print(f"[INFO] Accuracy={acc_rate*100:.2f}%, Error={error_rate*100:.2f}%")

        # --- 混同行列を保存 ---
        if save_confusion:
            cm_save_path = os.path.splitext(save_path)[0] + "_confusion.png"
            cm = confusion_matrix(y_true, y_pred, labels = np.unique(y_true))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix ({dataset_keyword})")
            plt.savefig(cm_save_path, bbox_inches="tight", dpi = 300)
            plt.close()
            print(f"[INFO] 混同行列を保存しました → {cm_save_path}")

    # ============================================================
    # テストデータ上の可視化(従来通り)
    # ============================================================
    else:
        # 可視化対象クラス (背景を除く)
        unique_classes = np.unique(pred_2d[pred_2d != background_label])
        # --- カラーマップ読み込み ---
        try:
            if dataset_keyword is not None and rs_csv_root is not None:
                color_dict, name_dict = load_colormap_from_csv(
                    dataset_keyword=dataset_keyword,
                    rs_csv_root=rs_csv_root
                )
                base_colors = {
                    cls: to_rgba(color_dict.get(cls, "#808080"))
                    for cls in unique_classes
                }
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

        # if overlay and X_rgb is not None:
            # base_img = (X_rgb - X_rgb.min()) / (X_rgb.max() - X_rgb.min() + 1e-8)
            # overlay_img = 0.6 * base_img + 0.4 * display[..., :3]
            # display = overlay_img

        plt.figure(figsize=(8,8))
        plt.imshow(display)
        plt.title(f"Test Prediction Map: {dataset_keyword}")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[INFO] テストデータ可視化を保存しました → {save_path}")

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

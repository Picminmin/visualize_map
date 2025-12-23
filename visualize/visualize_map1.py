import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# visualize_map.py の場所を基準にする
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RS_CSV_ROOT = os.path.join(os.path.dirname(THIS_DIR), "rs_csv")


# ============================================================
# 1) Dataset-specific remap (SalinasA)
#    SalinasA: {1,10,11,12,13,14} -> {1,2,3,4,5,6}
# ============================================================
def remap_salinasA_labels(labels, background_label=0):
    """
    SalinasA の元ラベルが [1,10,11,12,13,14] のケースを
    [1,2,3,4,5,6] に詰め直す。
    """
    labels = np.asarray(labels).copy()
    mapping = {1: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}

    mask = labels != background_label
    # 未知ラベルはそのまま残し（後段の検出で警告）、既知だけ変換
    for src, dst in mapping.items():
        labels[labels == src] = dst

    return labels


# ============================================================
# 2) CSV loader
# ============================================================
def load_colormap_from_csv(dataset_keyword, rs_csv_root=RS_CSV_ROOT):
    """
    rs_csv/<dataset_keyword>/ 内にある CSV を読み込み、
    class_id + color, class_id → name の辞書を返す。
    """
    csv_dir = os.path.join(rs_csv_root, dataset_keyword)
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"[ERROR] ディレクトリが存在しません: {csv_dir}")

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if len(csv_files) != 1:
        raise RuntimeError(
            f"[ERROR] {csv_dir} には CSV ファイルが１つだけ存在する必要があります。"
            f"見つかった数: {len(csv_files)}"
        )

    csv_path = os.path.join(csv_dir, csv_files[0])
    df = pd.read_csv(csv_path)

    # 並びを安定させる（CSVのclass_id順）
    df = df.sort_values("class_id").reset_index(drop=True)

    color_dict = dict(zip(df["class_id"], df["color"]))
    name_dict = dict(zip(df["class_id"], df["name"]))

    return color_dict, name_dict


# ============================================================
# 3) LabelNormalizer (general + warnings)
# ============================================================
class LabelNormalizer:
    """
    可視化のためにラベルを正規化して、色・名称・順序を統一する小クラス。

    - CSV の class_id 順を「正」として扱う（凡例の順序が安定）
    - データ側が SalinasA 形式 (1,10..14) の場合は自動で詰め直す
    - 「CSVにあるのにデータに無い」「データにあるのにCSVに無い」を検出して警告
    """

    def __init__(self, dataset_keyword=None, rs_csv_root=RS_CSV_ROOT, background_label=0):
        self.dataset_keyword = dataset_keyword
        self.rs_csv_root = rs_csv_root
        self.background_label = background_label

        self.color_dict = None
        self.name_dict = None
        self.csv_class_ids = None  # CSVに定義されたclass_id（昇順）

        if dataset_keyword is not None and rs_csv_root is not None:
            self.color_dict, self.name_dict = load_colormap_from_csv(
                dataset_keyword=dataset_keyword,
                rs_csv_root=rs_csv_root
            )
            self.csv_class_ids = np.array(sorted(self.color_dict.keys()), dtype=int)

    def _maybe_salinasA_remap(self, labels):
        if self.dataset_keyword is None:
            return labels
        if str(self.dataset_keyword).lower() != "salinasa":
            return labels

        labels = np.asarray(labels)
        non_bg = labels[labels != self.background_label]
        if non_bg.size == 0:
            return labels
        # SalinasAの典型(10..14が出る)なら詰め直し
        if np.any(np.isin(non_bg, [10, 11, 12, 13, 14])):
            return remap_salinasA_labels(labels, background_label=self.background_label)
        return labels

    def normalize(self, labels):
        """
        可視化用に labels を正規化する。
        - SalinasA の場合は 1,10..14 -> 1..6 に詰め直し
        - それ以外は「値自体は維持」でもよいが、CSV順で凡例を作るためここで整形を統一
        """
        labels = self._maybe_salinasA_remap(labels)
        return np.asarray(labels)

    def warn_missing(self, labels, context=""):
        """
        欠損クラス検出と警告表示
        - CSVに定義があるのにデータに出ない
        - データに出るのにCSVに定義がない（=灰色になり得る）
        """
        if self.csv_class_ids is None:
            return

        labels = np.asarray(labels)
        present = np.unique(labels[labels != self.background_label])

        missing_in_data = set(self.csv_class_ids.tolist()) - set(present.tolist())
        missing_in_csv = set(present.tolist()) - set(self.csv_class_ids.tolist())

        if missing_in_csv:
            print(f"[WARN]{context} CSV(colormap)に未定義の class_id が含まれます: {sorted(missing_in_csv)}")
        if missing_in_data:
            print(f"[INFO]{context} データ中に出現しない class_id (CSV定義済み): {sorted(missing_in_data)}")

    def iter_classes_for_legend(self, labels=None):
        """
        凡例/描画に使うクラス順を返す。
        - CSVがあればCSV順
        - なければlabelsに出てくる順（昇順）
        """
        if self.csv_class_ids is not None:
            return self.csv_class_ids.tolist()
        if labels is None:
            return []
        labels = np.asarray(labels)
        return np.unique(labels[labels != self.background_label]).tolist()

    def rgba_for_class(self, class_id, default="#808080"):
        """
        class_id -> RGBA
        """
        if self.color_dict is None:
            return to_rgba(default)
        return to_rgba(self.color_dict.get(int(class_id), default))

    def name_for_class(self, class_id):
        if self.name_dict is None:
            return f"Class {class_id}"
        return self.name_dict.get(int(class_id), f"Class {class_id}")


# ============================================================
# 4) visualize_train_test_split_map (unified)
# ============================================================
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
    os.makedirs(save_dir, exist_ok=True)

    normalizer = LabelNormalizer(
        dataset_keyword=dataset_keyword,
        rs_csv_root=rs_csv_root,
        background_label=background_label
    )

    # y を正規化（SalinasA含む）
    y_vis = normalizer.normalize(y)

    # 警告（欠損クラスなど）
    normalizer.warn_missing(y_vis, context="[train_test_split]")

    # 凡例順
    classes = normalizer.iter_classes_for_legend(y_vis)

    # base colors/labels
    base_colors = [normalizer.rgba_for_class(c) for c in classes]
    labels = [normalizer.name_for_class(c) for c in classes]

    # ==============================
    # (2) 空間オーバーレイ図
    # ==============================
    display = np.full((*y_vis.shape, 4), np.nan)
    for idx, cls in enumerate(classes):
        cls_mask = (y_vis == cls)
        display[cls_mask] = to_rgba(base_colors[idx], alpha=0.6)
        display[np.logical_and(cls_mask, train_mask)] = to_rgba(base_colors[idx], alpha=1.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(f"{title}\nTrain={np.sum(train_mask)}, Test={np.sum(test_mask)}", fontsize=12)
    plt.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=base_colors[i], markersize=10)
        for i in range(len(classes))
    ]
    plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)

    overlay_path = os.path.join(save_dir, f"{dataset_keyword}_spatial_split_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ==============================
    # (3) クラスごとのサンプル数棒グラフ
    # ==============================
    train_counts = [np.sum(np.logical_and(y_vis == cls, train_mask)) for cls in classes]
    test_counts = [np.sum(np.logical_and(y_vis == cls, test_mask)) for cls in classes]

    x = np.arange(len(classes))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, train_counts, bar_width, label="Train")
    plt.bar(x + bar_width / 2, test_counts, bar_width, label="Test")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Sample Count")
    plt.title(f"{dataset_keyword}: Class-wise Train/Test Distribution")
    plt.legend()

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


# ============================================================
# 5) visualize_iteration_map (unified)
# ============================================================
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
    boundary=None,
    boundary_color=(1.0, 0.3, 0.3, 0.4),
    seed_mask=None
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W = image_shape

    normalizer = LabelNormalizer(
        dataset_keyword=dataset_keyword,
        rs_csv_root=rs_csv_root,
        background_label=background_label
    )

    y_vis = normalizer.normalize(y)
    expand_vis = normalizer.normalize(expand_label)

    normalizer.warn_missing(y_vis, context="[iteration_map]")

    expand2d = expand_vis.reshape(H, W)

    classes = normalizer.iter_classes_for_legend(y_vis)

    display = np.ones((H, W, 4), dtype=float)  # 白背景
    for cls in classes:
        cls_mask = (expand2d == cls)
        display[cls_mask] = normalizer.rgba_for_class(cls)

    # 未ラベル灰色
    mask_U = np.zeros(H * W, dtype=bool)
    mask_U[U_index] = True
    mask_U = mask_U.reshape(H, W)
    display[mask_U] = (0.8, 0.8, 0.8, 1.0)

    # boundary overlay
    if boundary is not None:
        if boundary.shape != (H, W):
            raise ValueError(f"boundary shape mismatch: expected {(H, W)}, got {boundary.shape}")
        boundary_mask = boundary.astype(bool)
        for c in range(3):
            display[..., c] = np.where(
                boundary_mask,
                boundary_color[c]*boundary_color[3] + display[..., c] * (1 - boundary_color[3]),
                display[..., c]
            )
        display[..., 3] = np.where(
            boundary_mask,
            np.maximum(display[..., 3], boundary_color[3]),
            display[..., 3]
        )

    # seed candidates overlay
    if seed_mask is not None:
        seed_mask = seed_mask.reshape(H, W).astype(bool)
        seed_color = (0.3, 0.6, 1.0, 0.2)
        if np.any(seed_mask):
            for c in range(3):
                display[..., c] = np.where(
                    seed_mask,
                    seed_color[c]*seed_color[3] + display[..., c] * (1 - seed_color[3]),
                    display[..., c]
                )
            display[..., 3] = np.where(
                seed_mask,
                np.maximum(display[..., 3], seed_color[3]),
                display[..., 3]
            )

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Visualization saved → {save_path}")


# ============================================================
# 6) visualize_iteration_map_v2 (unified)
# ============================================================
def visualize_iteration_map_v2(
    y,                           # ground truth (flat)
    expand_label,                # current expanded labels (flat)
    L_index, U_index,            # labeled/unlabeled indices
    image_shape,                 # (H, W)
    save_path,                   # save
    dataset_keyword=None,        # for colormap
    boundary=None,               # edge mask H*W
    seed_mask=None,              # 孤立領域全体のマスク (H*W,)
    seed_low_mask=None,          # low-conf seed candidate mask H*W
    seed_high_mask=None,         # high-conf seed candidate mask H*W
    rs_csv_root=RS_CSV_ROOT,
    title="Iteration Map (v2)",
    background_label=0,
    boundary_color=(1.0, 0.3, 0.3, 0.4),  # red for edges
    seed_low_color=(0.3, 0.6, 1.0, 0.2),  # blue for low/mid confidence seeds
    seed_high_color=(1.0, 1.0, 0.3, 0.7)  # yellow for high confidence seeds
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W = image_shape

    normalizer = LabelNormalizer(
        dataset_keyword=dataset_keyword,
        rs_csv_root=rs_csv_root,
        background_label=background_label
    )

    y_vis = normalizer.normalize(y)
    expand_vis = normalizer.normalize(expand_label)

    # 欠損検出
    normalizer.warn_missing(y_vis, context="[iteration_map_v2]")

    expand2d = expand_vis.reshape(H, W)
    classes = normalizer.iter_classes_for_legend(y_vis)

    # 初期表示 (白背景)
    display = np.ones((H, W, 4), dtype=float)

    # クラス色
    for cls in classes:
        cls_mask = (expand2d == cls)
        display[cls_mask] = normalizer.rgba_for_class(cls)

    # 未ラベル領域 (U_index) は灰色
    unlabeled_mask = np.zeros(H * W, dtype=bool)
    unlabeled_mask[U_index] = True
    unlabeled_mask = unlabeled_mask.reshape(H, W)
    display[unlabeled_mask] = (0.8, 0.8, 0.8, 1.0)

    # boundary overlay
    if boundary is not None:
        boundary_mask = boundary.astype(bool)
        for c in range(3):
            display[..., c] = np.where(
                boundary_mask,
                boundary_color[c]*boundary_color[3] + display[..., c] * (1 - boundary_color[3]),
                display[..., c]
            )
        display[..., 3] = np.where(
            boundary_mask,
            np.maximum(display[..., 3], boundary_color[3]),
            display[..., 3]
        )

    # seed_high_mask overlay (薄黄色)
    if seed_high_mask is not None:
        sm = seed_high_mask.reshape(H, W).astype(bool)
        if np.any(sm):
            for c in range(3):
                display[..., c] = np.where(
                    sm,
                    seed_high_color[c]*seed_high_color[3] + display[..., c] * (1 - seed_high_color[3]),
                    display[..., c]
                )
            display[..., 3] = np.where(
                sm,
                np.maximum(display[..., 3], seed_high_color[3]),
                display[..., 3]
            )

    # seed_low_mask overlay (薄青)
    if seed_low_mask is not None:
        sm = seed_low_mask.reshape(H, W).astype(bool)
        if np.any(sm):
            for c in range(3):
                display[..., c] = np.where(
                    sm,
                    seed_low_color[c]*seed_low_color[3] + display[..., c] * (1 - seed_low_color[3]),
                    display[..., c]
                )
            display[..., 3] = np.where(
                sm,
                np.maximum(display[..., 3], seed_low_color[3]),
                display[..., 3]
            )

    # seed_mask only fallback
    if (seed_low_mask is None) and (seed_high_mask is None) and (seed_mask is not None):
        sm = seed_mask.reshape(H, W).astype(bool)
        seed_color = (0.3, 0.6, 1.0, 0.2)
        if np.any(sm):
            for c in range(3):
                display[..., c] = np.where(
                    sm,
                    seed_color[c]*seed_color[3] + display[..., c] * (1 - seed_color[3]),
                    display[..., c]
                )
            display[..., 3] = np.where(
                sm,
                np.maximum(display[..., 3], seed_color[3]),
                display[..., 3]
            )

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Visualization saved → {save_path}")


# ============================================================
# 7) visualize_prediction_map (unified for SalinasA too)
# ============================================================
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
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W = image_shape

    normalizer = LabelNormalizer(
        dataset_keyword=dataset_keyword,
        rs_csv_root=rs_csv_root,
        background_label=background_label
    )

    # --- 1. ベース画像を背景ラベルで初期化 ---
    pred_full = np.full(H * W, background_label, dtype=int)
    pred_full[pred_index] = y_pred
    pred_full = normalizer.normalize(pred_full)  # ★統一：SalinasAもここで正規化
    pred_2d = pred_full.reshape(H, W)

    if y_true is not None:
        # 教師データ上の可視化:正解(青)/誤分類(赤)
        true_full = np.full(H * W, background_label, dtype=int)
        true_full[pred_index] = y_true
        true_full = normalizer.normalize(true_full)  # ★統一
        true_2d = true_full.reshape(H, W)

        correct_mask = (pred_2d == true_2d) & (true_2d != background_label)
        wrong_mask = (pred_2d != true_2d) & (true_2d != background_label)

        display = np.ones((H, W, 3))
        display[correct_mask] = [0.2, 0.4, 1.0]
        display[wrong_mask] = [1.0, 0.2, 0.2]

        plt.figure(figsize=(8, 8))
        plt.imshow(display)
        plt.title(f"Train Prediction Map {dataset_keyword}")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[INFO] Train visualization saved → {save_path}")

        if save_confusion:
            cm_save_path = os.path.splitext(save_path)[0] + "_confusion.png"
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix ({dataset_keyword})")
            plt.savefig(cm_save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"[INFO] Confusion matrix saved → {cm_save_path}")

        return

    # テストデータ上の可視化（従来通り＋正規化）
    normalizer.warn_missing(pred_full, context="[prediction_map]")

    classes = normalizer.iter_classes_for_legend(pred_full)

    display = np.ones((H, W, 4), dtype=float)
    for cls in classes:
        cls_mask = (pred_2d == cls)
        if np.any(cls_mask):
            display[cls_mask] = normalizer.rgba_for_class(cls)

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(f"Test Prediction Map: {dataset_keyword}")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] テストデータ可視化を保存しました → {save_path}")

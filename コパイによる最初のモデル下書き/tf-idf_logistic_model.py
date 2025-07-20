"""
MAP - Charting Student Math Misunderstandings
機械学習モデル開発スクリプト

コンペ概要:
- 学生の数学の説明文から誤概念を予測
- 評価指標: MAP@3
- ラベル: True/False_Correct/Neither/Misconception
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
import warnings

warnings.filterwarnings("ignore")

# 日本語フォント設定
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "Yu Gothic",
    "Hiragino Sans",
    "Takao",
    "IPAexGothic",
    "IPAPGothic",
    "VL PGothic",
    "Noto Sans CJK JP",
]


def load_data():
    """データの読み込み"""
    print("=" * 50)
    print("データ読み込み中...")

    train_df = pd.read_csv("map_data/train.csv")
    test_df = pd.read_csv("map_data/test.csv")
    sample_submission = pd.read_csv("map_data/sample_submission.csv")

    print(f"訓練データ: {train_df.shape}")
    print(f"テストデータ: {test_df.shape}")
    print(f"サンプル提出: {sample_submission.shape}")

    return train_df, test_df, sample_submission


def analyze_target_distribution(train_df):
    """ターゲット変数の分析"""
    print("\n" + "=" * 50)
    print("ターゲット変数分析")
    print("=" * 50)

    # Category分布
    print("Category分布:")
    category_counts = train_df["Category"].value_counts()
    print(category_counts)
    print(f"パーセンテージ:\n{(category_counts / len(train_df) * 100).round(2)}")

    # Misconception分布（NAを除く）
    print("\nMisconception分布（NA除く）:")
    misconception_counts = train_df[train_df["Misconception"] != "NA"][
        "Misconception"
    ].value_counts()
    print(f"ユニークな誤概念数: {len(misconception_counts)}")
    print("上位10の誤概念:")
    print(misconception_counts.head(10))

    # Category:Misconceptionの組み合わせ作成
    train_df["target"] = train_df["Category"] + ":" + train_df["Misconception"]
    target_counts = train_df["target"].value_counts()
    print(f"\nユニークなCategory:Misconception組み合わせ: {len(target_counts)}")
    print("上位10の組み合わせ:")
    print(target_counts.head(10))

    # NaN値の確認と除去
    print(f"\nNaN値の確認:")
    print(f"Category NaN数: {train_df['Category'].isna().sum()}")
    print(f"Misconception NaN数: {train_df['Misconception'].isna().sum()}")
    print(f"target NaN数: {train_df['target'].isna().sum()}")

    # NaN値を含む行を除去
    before_len = len(train_df)
    train_df = train_df.dropna(subset=["Category", "Misconception", "target"])
    after_len = len(train_df)
    print(f"NaN除去: {before_len} -> {after_len} ({before_len - after_len}行削除)")

    return train_df


def preprocess_text(text):
    """テキスト前処理"""
    if pd.isna(text):
        return ""

    # 小文字化
    text = text.lower()

    # 数字を統一
    text = re.sub(r"\d+", "[NUM]", text)

    # 特殊文字削除（基本的な句読点は残す）
    text = re.sub(r"[^\w\s\.,!?]", " ", text)

    # 複数スペースを単一スペースに
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_text_features(train_df, test_df):
    """テキスト特徴量の作成"""
    print("\n" + "=" * 50)
    print("テキスト特徴量作成")
    print("=" * 50)

    # 基本的なテキスト統計
    def text_stats(df):
        df["explanation_length"] = df["StudentExplanation"].fillna("").str.len()
        df["explanation_words"] = (
            df["StudentExplanation"].fillna("").str.split().str.len()
        )
        df["explanation_sentences"] = (
            df["StudentExplanation"].fillna("").str.count(r"[.!?]+") + 1
        )

        # 数字の出現回数
        df["num_count"] = df["StudentExplanation"].fillna("").str.count(r"\d")

        # 数学関連キーワード
        math_keywords = [
            "fraction",
            "decimal",
            "percent",
            "multiply",
            "divide",
            "add",
            "subtract",
            "numerator",
            "denominator",
            "equal",
            "greater",
            "less",
            "because",
        ]
        for keyword in math_keywords:
            df[f"has_{keyword}"] = (
                df["StudentExplanation"]
                .fillna("")
                .str.lower()
                .str.contains(keyword)
                .astype(int)
            )

        return df

    train_df = text_stats(train_df)
    test_df = text_stats(test_df)

    # テキスト前処理
    train_df["processed_explanation"] = (
        train_df["StudentExplanation"].fillna("").apply(preprocess_text)
    )
    test_df["processed_explanation"] = (
        test_df["StudentExplanation"].fillna("").apply(preprocess_text)
    )

    print("テキスト特徴量作成完了")
    print(
        f"追加された特徴量数: {len([col for col in train_df.columns if col.startswith('has_') or col.endswith('_length') or col.endswith('_words') or col.endswith('_sentences') or col.endswith('_count')])}"
    )

    return train_df, test_df


def create_tfidf_features(train_df, test_df, max_features=5000):
    """TF-IDF特徴量の作成"""
    print(f"\nTF-IDF特徴量作成 (max_features={max_features})")

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    # 全てのテキストでfit
    all_texts = pd.concat(
        [train_df["processed_explanation"], test_df["processed_explanation"]]
    )
    tfidf.fit(all_texts)

    # 変換
    train_tfidf = tfidf.transform(train_df["processed_explanation"])
    test_tfidf = tfidf.transform(test_df["processed_explanation"])

    print(f"TF-IDF特徴量形状: {train_tfidf.shape}")

    return train_tfidf, test_tfidf, tfidf


def baseline_model(train_df, train_tfidf):
    """ベースラインモデルの構築"""
    print("\n" + "=" * 50)
    print("ベースラインモデル構築")
    print("=" * 50)

    # 数値特徴量の選択
    numeric_features = [
        "explanation_length",
        "explanation_words",
        "explanation_sentences",
        "num_count",
    ]
    numeric_features += [col for col in train_df.columns if col.startswith("has_")]

    X_numeric = train_df[numeric_features].fillna(0)

    # TF-IDFと数値特徴量を結合
    from scipy.sparse import hstack

    X_combined = hstack([train_tfidf, X_numeric.values])

    y = train_df["target"]

    print(f"特徴量形状: {X_combined.shape}")
    print(f"ターゲットのユニーク数: {y.nunique()}")

    # ターゲット変数の確認とクリーンアップ
    print(f"ターゲット変数のNaN数: {y.isna().sum()}")

    # NaN値がある場合は除去
    if y.isna().sum() > 0:
        valid_indices = ~y.isna()
        X_combined = X_combined[valid_indices]
        y = y[valid_indices]
        print(f"NaN値除去後の形状: {X_combined.shape}")

    # 訓練・検証分割（少数クラスの場合はstratifyなし）
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Stratified split使用")
    except ValueError:
        # 少数クラスがある場合はstratifyなしで分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )
        print("Regular split使用（少数クラスのため）")

    # ロジスティック回帰
    print("ロジスティック回帰モデル訓練中...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)

    # 予測
    y_pred = lr_model.predict(X_val)

    # 評価
    print("\n検証結果:")
    print(classification_report(y_val, y_pred))

    return lr_model, X_train, X_val, y_train, y_val


def calculate_map3(y_true, y_pred_proba, model_classes):
    """MAP@3の計算"""
    map_scores = []

    for i, true_label in enumerate(y_true):
        # 予測確率の上位3つを取得
        top3_indices = np.argsort(y_pred_proba[i])[::-1][:3]
        top3_labels = [model_classes[idx] for idx in top3_indices]

        # MAP計算
        score = 0.0
        for j, pred_label in enumerate(top3_labels):
            if pred_label == true_label:
                score = 1.0 / (j + 1)
                break
        map_scores.append(score)

    return np.mean(map_scores)


def evaluate_map3(model, X_val, y_val):
    """MAP@3での評価"""
    print("\nMAP@3評価中...")

    # 予測確率
    y_pred_proba = model.predict_proba(X_val)

    # MAP@3計算
    map3_score = calculate_map3(y_val.values, y_pred_proba, model.classes_)

    print(f"MAP@3スコア: {map3_score:.4f}")

    return map3_score


def main():
    """メイン実行関数"""
    print("MAP競技 機械学習モデル開発開始")
    print("=" * 50)

    # データ読み込み
    train_df, test_df, sample_submission = load_data()

    # ターゲット分析
    train_df = analyze_target_distribution(train_df)

    # 特徴量作成
    train_df, test_df = create_text_features(train_df, test_df)

    # TF-IDF特徴量
    train_tfidf, test_tfidf, tfidf_vectorizer = create_tfidf_features(train_df, test_df)

    # ベースラインモデル
    model, X_train, X_val, y_train, y_val = baseline_model(train_df, train_tfidf)

    # MAP@3評価
    map3_score = evaluate_map3(model, X_val, y_val)

    print("\n" + "=" * 50)
    print("ベースラインモデル完成!")
    print(f"MAP@3スコア: {map3_score:.4f}")
    print("=" * 50)

    return train_df, test_df, model, tfidf_vectorizer, map3_score


if __name__ == "__main__":
    # Pythonパスとライブラリ確認
    import sys

    print(f"Python実行パス: {sys.executable}")
    print(f"作業ディレクトリ: {sys.path[0] if sys.path else 'Unknown'}")

    # メイン実行
    train_df, test_df, model, tfidf_vectorizer, map3_score = main()

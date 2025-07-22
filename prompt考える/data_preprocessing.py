"""
データの前処理スクリプト
ラベリングとプロンプトの調整を行う
"""

import pandas as pd
import os


def preprocess_data():
    """
    train.csvとtest.csvのデータを前処理する

    ラベリング形式: Category:Misconception (例: False_Misconception:WNB, True_Correct:NA)
    プロンプト形式: QuestionText + MC_Answer + StudentExplanation を結合
    """

    # データファイルのパス（一つ上のディレクトリからの相対パス）
    train_path = "../map_data/train.csv"
    test_path = "../map_data/test.csv"
    sample_submission_path = "../map_data/sample_submission.csv"

    # 出力ファイルのパス（現在のディレクトリ内）
    processed_train_path = "processed_train.csv"
    processed_test_path = "processed_test.csv"

    print("データの前処理を開始します...")

    # 1. 訓練データの処理
    if os.path.exists(train_path):
        print(f"訓練データを読み込み中: {train_path}")
        train_df = pd.read_csv(train_path)

        # プロンプトの作成: QuestionText + MC_Answer + StudentExplanation
        train_df["prompt"] = (
            "Question: "
            + train_df["QuestionText"].astype(str)
            + "\nCorrect Answer: "
            + train_df["MC_Answer"].astype(str)
            + "\nStudent Explanation: "
            + train_df["StudentExplanation"].astype(str)
        )

        # ラベルの作成: Category:Misconception
        # nanをNAに置換
        train_df["Misconception"] = train_df["Misconception"].fillna("NA")
        train_df["label"] = (
            train_df["Category"].astype(str)
            + ":"
            + train_df["Misconception"].astype(str)
        )

        # 必要な列のみを保持
        processed_train = train_df[["row_id", "prompt", "label"]].copy()

        # 保存
        processed_train.to_csv(processed_train_path, index=False)
        print(f"処理済み訓練データを保存: {processed_train_path}")
        print(f"訓練データサンプル数: {len(processed_train)}")

        # ラベルの分布を確認
        print("\nラベル分布:")
        print(processed_train["label"].value_counts().head(10))

    else:
        print(f"訓練データが見つかりません: {train_path}")

    # 2. テストデータの処理
    if os.path.exists(test_path):
        print(f"\nテストデータを読み込み中: {test_path}")
        test_df = pd.read_csv(test_path)

        # プロンプトの作成: QuestionText + MC_Answer + StudentExplanation
        test_df["prompt"] = (
            "Question: "
            + test_df["QuestionText"].astype(str)
            + "\nCorrect Answer: "
            + test_df["MC_Answer"].astype(str)
            + "\nStudent Explanation: "
            + test_df["StudentExplanation"].astype(str)
        )

        # 必要な列のみを保持
        processed_test = test_df[["row_id", "prompt"]].copy()

        # 保存
        processed_test.to_csv(processed_test_path, index=False)
        print(f"処理済みテストデータを保存: {processed_test_path}")
        print(f"テストデータサンプル数: {len(processed_test)}")

    else:
        print(f"テストデータが見つかりません: {test_path}")

    # 3. サンプル提出ファイルの形式を確認・更新
    if os.path.exists(sample_submission_path):
        print(f"\nサンプル提出ファイルを確認中: {sample_submission_path}")
        sample_df = pd.read_csv(sample_submission_path)
        print("現在の提出形式:")
        print(sample_df.head())

        # 提出形式の例を作成
        sample_df["Category:Misconception"] = sample_df[
            "Category:Misconception"
        ].fillna("False_Misconception:Incomplete")
        print("\n調整後の提出形式例:")
        print(sample_df.head())

        # 更新された形式で保存（現在のディレクトリ内）
        updated_sample_path = "updated_sample_submission.csv"
        sample_df.to_csv(updated_sample_path, index=False)
        print(f"更新済みサンプル提出ファイルを保存: {updated_sample_path}")

    print("\nデータ前処理が完了しました!")


def show_examples():
    """処理済みデータの例を表示"""
    processed_train_path = "processed_train.csv"

    if os.path.exists(processed_train_path):
        df = pd.read_csv(processed_train_path)

        print("=== 処理済みデータの例 ===")
        for i in range(min(3, len(df))):
            print(f"\n--- 例 {i+1} ---")
            print(f"Row ID: {df.iloc[i]['row_id']}")
            print(f"Label: {df.iloc[i]['label']}")
            print(f"Prompt:\n{df.iloc[i]['prompt']}")
            print("-" * 50)


if __name__ == "__main__":
    preprocess_data()
    print("\n" + "=" * 60)
    show_examples()

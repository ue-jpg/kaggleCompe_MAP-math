"""
正確な全ラベル付きのcompactプロンプト
実データから取得した65個の全ラベルを含む
"""

import pandas as pd


def get_actual_labels_from_data():
    """実際のデータから全ラベルを取得"""
    try:
        # 既存の処理済みデータから取得
        df = pd.read_csv("processed_train_instructional.csv")
        return sorted(df["label"].unique())
    except FileNotFoundError:
        # 元データから生成
        df = pd.read_csv("../map_data/train.csv")
        labels = (
            df["Category"].astype(str)
            + ":"
            + df["Misconception"].fillna("NA").astype(str)
        )
        return sorted(labels.unique())


def get_corrected_compact_prompt(question, answer, explanation):
    """修正版コンパクトプロンプト - False_Correct:NAを含む全ラベル付き"""

    all_labels = get_actual_labels_from_data()
    labels_text = "\n".join([f"- {label}" for label in all_labels])

    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons  
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

TASK: Classify this student's response using EXACTLY ONE of these {len(all_labels)} labels:

{labels_text}

Classification:"""

    return prompt


def create_final_instructional_compact_dataset():
    """最終版のinstructional compactデータセットを生成"""

    # 元データを読み込み（一部のみ処理）
    train_path = "../map_data/train.csv"
    train_df = pd.read_csv(train_path)

    # 処理時間短縮のため最初の100行のみ処理
    # 全データ処理時は以下の行をコメントアウト
    train_df = train_df.head(100)

    print("最終版instructional compactプロンプトでデータセット生成中...")
    print(f"処理対象: {len(train_df)} samples (時間短縮のため)")
    print(f"全ラベル数: {len(get_actual_labels_from_data())}")

    # プロンプトの生成
    compact_prompts = []
    for _, row in train_df.iterrows():
        prompt = get_corrected_compact_prompt(
            str(row["QuestionText"]),
            str(row["MC_Answer"]),
            str(row["StudentExplanation"]),
        )
        compact_prompts.append(prompt)

    # データセット作成
    final_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"],
            "prompt": compact_prompts,
            "label": train_df["Category"].astype(str)
            + ":"
            + train_df["Misconception"].fillna("NA").astype(str),
        }
    )

    # 保存
    output_file = "processed_train_instructional_compact_final_sample.csv"
    final_df.to_csv(output_file, index=False)
    print(f"最終版保存完了: {output_file} ({len(final_df)} samples)")

    # ラベル確認
    actual_labels = sorted(final_df["label"].unique())
    print(f"\n実際のラベル数: {len(actual_labels)}")
    print("含まれているラベル（最初の10個）:")
    for i, label in enumerate(actual_labels[:10]):
        print(f"  {i+1}. {label}")

    # False_Correctが含まれているか確認
    false_correct_labels = [
        label for label in actual_labels if "False_Correct" in label
    ]
    print(f"\nFalse_Correct関連ラベル: {false_correct_labels}")

    # プロンプト例表示
    print(f"\n=== 最終版プロンプト例 ===")
    print(f"Label: {final_df.iloc[0]['label']}")
    print(f"Prompt length: {len(final_df.iloc[0]['prompt'])} 文字")
    print(f"Prompt preview:\n{final_df.iloc[0]['prompt'][:400]}...")

    return final_df


def test_sample_prompts():
    """サンプルプロンプトをテスト"""
    sample_question = "What is 1/2 + 1/3?"
    sample_answer = "5/6"
    sample_explanation = "I added 1+1=2 and 2+3=5, so the answer is 2/5"

    prompt = get_corrected_compact_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print("=== サンプルプロンプト ===")
    print(f"Length: {len(prompt)} 文字")
    print(prompt)


if __name__ == "__main__":
    # 1. ラベル確認
    labels = get_actual_labels_from_data()
    print(f"実データから取得した全ラベル数: {len(labels)}")
    print("\n全ラベル一覧:")
    for i, label in enumerate(labels, 1):
        print(f"{i:2d}. {label}")

    # 2. 最終データセット生成
    print("\n" + "=" * 60)
    create_final_instructional_compact_dataset()

    # 3. サンプルテスト
    print("\n" + "=" * 60)
    test_sample_prompts()

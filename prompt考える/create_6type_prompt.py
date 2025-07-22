"""
改良版データ前処理スクリプト
複数のプロンプトテンプレートをテストできる版
"""

import pandas as pd
import os
from prompt_templates import get_prompt, PROMPT_TEMPLATES


def preprocess_data_with_templates():
    """
    複数のプロンプトテンプレートでデータを前処理する
    """

    # データファイルのパス
    train_path = "../map_data/train.csv"
    test_path = "../map_data/test.csv"

    print("改良版データ前処理を開始します...")

    if not os.path.exists(train_path):
        print(f"訓練データが見つかりません: {train_path}")
        return

    print(f"訓練データを読み込み中: {train_path}")
    train_df = pd.read_csv(train_path)

    # 各プロンプトテンプレートでデータを生成
    for template_name in PROMPT_TEMPLATES.keys():
        print(f"\n=== {template_name.upper()} テンプレートで処理中 ===")

        # プロンプトの作成
        prompts = []
        for _, row in train_df.iterrows():
            prompt = get_prompt(
                template_name,
                str(row["QuestionText"]),
                str(row["MC_Answer"]),
                str(row["StudentExplanation"]),
            )
            prompts.append(prompt)

        # データフレームの作成
        processed_df = pd.DataFrame(
            {
                "row_id": train_df["row_id"],
                "prompt": prompts,
                "label": train_df["Category"].astype(str)
                + ":"
                + train_df["Misconception"].fillna("NA").astype(str),
            }
        )

        # 保存（現在のディレクトリ内）
        output_path = f"processed_train_{template_name}.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"保存完了: {output_path}")
        print(f"サンプル数: {len(processed_df)}")

        # 最初の例を表示
        print(f"\n{template_name} プロンプト例:")
        print("-" * 80)
        print(f"Label: {processed_df.iloc[0]['label']}")
        print(f"Prompt:\n{processed_df.iloc[0]['prompt']}")
        print("-" * 80)


def create_prompt_comparison():
    """
    異なるプロンプトテンプレートの比較用サンプルを作成
    """
    train_path = "../map_data/train.csv"

    if not os.path.exists(train_path):
        return

    train_df = pd.read_csv(train_path)

    # 異なるラベルの代表例を取得
    sample_indices = []
    unique_labels = train_df["Category"].unique()

    for label in unique_labels[:4]:  # 最初の4つのカテゴリー
        sample_idx = train_df[train_df["Category"] == label].index[0]
        sample_indices.append(sample_idx)

    print("\n" + "=" * 100)
    print("プロンプトテンプレート比較")
    print("=" * 100)

    for i, idx in enumerate(sample_indices):
        row = train_df.iloc[idx]
        question = str(row["QuestionText"])
        answer = str(row["MC_Answer"])
        explanation = str(row["StudentExplanation"])
        label = f"{row['Category']}:{row['Misconception'] if pd.notna(row['Misconception']) else 'NA'}"

        print(f"\n【サンプル {i+1}】 Label: {label}")
        print("Question:", question[:100] + "..." if len(question) > 100 else question)
        print(
            "Student Explanation:",
            explanation[:100] + "..." if len(explanation) > 100 else explanation,
        )
        print("\n" + "-" * 50)

        for template_name in ["basic", "instructional", "few_shot"]:
            prompt = get_prompt(template_name, question, answer, explanation)
            print(f"\n【{template_name.upper()}】")
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
            print("-" * 30)


def recommend_best_template():
    """
    用途別推奨テンプレートの説明
    """
    print("\n" + "=" * 80)
    print("プロンプトテンプレート推奨用途")
    print("=" * 80)

    recommendations = {
        "basic": "シンプルで高速。小規模テストや基準線として最適",
        "instructional": "明確な指示で高精度。本格運用におすすめ",
        "step_by_step": "複雑な推論が必要な場合。精度重視",
        "few_shot": "例示学習でパフォーマンス向上。中程度のデータ量",
        "structured": "分析プロセスを重視。解釈可能性が重要な場合",
        "contextual": "教育的文脈を重視。実際の教育現場での使用",
    }

    for template, description in recommendations.items():
        print(f"• {template:12} : {description}")

    print(f"\n推奨: まず 'instructional' と 'few_shot' を試してください")


if __name__ == "__main__":
    # 1. すべてのテンプレートでデータを生成
    preprocess_data_with_templates()

    # 2. プロンプト比較表示
    create_prompt_comparison()

    # 3. 推奨事項表示
    recommend_best_template()

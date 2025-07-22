"""
改良版instructionalプロンプトでデータセットを生成
全ラベル付きバージョン
"""

import pandas as pd
from enhanced_prompt_templates import (
    get_enhanced_instructional_prompt,
    get_compact_enhanced_prompt,
)


def create_enhanced_dataset():
    """改良版プロンプトでデータセットを生成"""

    # 元データを読み込み
    train_path = "../map_data/train.csv"
    train_df = pd.read_csv(train_path)

    print("改良版instructionalプロンプトでデータセット生成中...")

    # 詳細版プロンプトの生成
    print("詳細版プロンプト生成中...")
    detailed_prompts = []
    for _, row in train_df.iterrows():
        prompt = get_enhanced_instructional_prompt(
            str(row["QuestionText"]),
            str(row["MC_Answer"]),
            str(row["StudentExplanation"]),
        )
        detailed_prompts.append(prompt)

    # 詳細版データセット作成
    detailed_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"],
            "prompt": detailed_prompts,
            "label": train_df["Category"].astype(str)
            + ":"
            + train_df["Misconception"].fillna("NA").astype(str),
        }
    )

    detailed_df.to_csv("processed_train_instructional_detailed.csv", index=False)
    print(
        f"詳細版保存完了: processed_train_instructional_detailed.csv ({len(detailed_df)} samples)"
    )

    # コンパクト版プロンプトの生成
    print("コンパクト版プロンプト生成中...")
    compact_prompts = []
    for _, row in train_df.iterrows():
        prompt = get_compact_enhanced_prompt(
            str(row["QuestionText"]),
            str(row["MC_Answer"]),
            str(row["StudentExplanation"]),
        )
        compact_prompts.append(prompt)

    # コンパクト版データセット作成
    compact_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"],
            "prompt": compact_prompts,
            "label": train_df["Category"].astype(str)
            + ":"
            + train_df["Misconception"].fillna("NA").astype(str),
        }
    )

    compact_df.to_csv("processed_train_instructional_compact.csv", index=False)
    print(
        f"コンパクト版保存完了: processed_train_instructional_compact.csv ({len(compact_df)} samples)"
    )

    # サンプル表示
    print("\n=== 詳細版プロンプト例 ===")
    print(f"Label: {detailed_df.iloc[0]['label']}")
    print(f"Prompt:\n{detailed_df.iloc[0]['prompt'][:300]}...")

    print("\n=== コンパクト版プロンプト例 ===")
    print(f"Label: {compact_df.iloc[0]['label']}")
    print(f"Prompt:\n{compact_df.iloc[0]['prompt'][:300]}...")


def compare_prompt_lengths():
    """プロンプトの長さを比較"""
    try:
        # 既存のinstructionalと新版を比較
        original_df = pd.read_csv("processed_train_instructional.csv")
        detailed_df = pd.read_csv("processed_train_instructional_detailed.csv")
        compact_df = pd.read_csv("processed_train_instructional_compact.csv")

        print("\n=== プロンプト長比較 ===")
        print(
            f"元のinstructional平均長: {original_df['prompt'].str.len().mean():.0f}文字"
        )
        print(f"詳細版平均長: {detailed_df['prompt'].str.len().mean():.0f}文字")
        print(f"コンパクト版平均長: {compact_df['prompt'].str.len().mean():.0f}文字")

        print("\n推奨:")
        print("- モデルのコンテキスト制限が厳しい場合: コンパクト版")
        print("- 高精度が必要でコンテキストに余裕がある場合: 詳細版")

    except FileNotFoundError:
        print("比較用ファイルが見つかりません")


if __name__ == "__main__":
    create_enhanced_dataset()
    compare_prompt_lengths()

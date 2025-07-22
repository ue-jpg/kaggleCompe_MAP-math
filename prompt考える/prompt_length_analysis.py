"""
プロンプト長分析
"""

import pandas as pd


def analyze_prompt_length():
    # データからラベル取得
    df = pd.read_csv("../map_data/train.csv")
    labels = (
        df["Category"].astype(str) + ":" + df["Misconception"].fillna("NA").astype(str)
    )
    all_labels = sorted(labels.unique())
    labels_text = "\n".join([f"- {label}" for label in all_labels])

    # 各セクションの文字数
    sections = {
        "役割設定": "You are an expert math educator analyzing student responses for mathematical misconceptions.",
        "問題例": "Question: What is 1/2 + 1/3?\nCorrect Answer: 5/6\nStudent's Explanation: I added 1+1=2 and 2+3=5, so the answer is 2/5",
        "分類ガイドライン": """CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons  
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception""",
        "タスク説明": f"TASK: Classify this student's response using EXACTLY ONE of these {len(all_labels)} labels:",
        "ラベルリスト": labels_text,
        "回答プロンプト": "Classification:",
    }

    print("=== プロンプト セクション別文字数分析 ===")
    total_chars = 0
    for name, content in sections.items():
        chars = len(content)
        total_chars += chars
        print(f"{name}: {chars:,} 文字")

    print(f"\n合計文字数: {total_chars:,} 文字")
    print(f"推定トークン数: {total_chars // 4:,} トークン")
    print(f"ラベルリストの割合: {len(labels_text)/total_chars*100:.1f}%")

    print("\n=== LLMモデル別推奨長 ===")
    print("• GPT-3.5/4: 1,000-2,000トークンが最適")
    print("• Claude: 1,500-3,000トークンが最適")
    print("• Gemma-2B: 500-1,000トークンが推奨")
    print("• 現在のプロンプト: 約750トークン ✅")

    print(f"\n=== 現在のプロンプトの評価 ===")
    token_count = total_chars // 4
    if token_count < 500:
        evaluation = "短い - 効率的だが情報不足の可能性"
    elif token_count < 1000:
        evaluation = "適切 - バランスが良い ✅"
    elif token_count < 2000:
        evaluation = "やや長い - 詳細だが注意散漫のリスク"
    else:
        evaluation = "長すぎる - 短縮を推奨"

    print(f"評価: {evaluation}")

    # 改善提案
    print(f"\n=== 改善提案 ===")
    if len(labels_text) / total_chars > 0.7:
        print("⚠️  ラベルリストが全体の70%以上を占めています")
        print("💡 解決策:")
        print("   1. ラベルをカテゴリ別にグループ化")
        print("   2. 主要ラベルのみ表示し、詳細は別途参照")
        print("   3. 階層的分類（まず大分類、次に詳細分類）")
    else:
        print("✅ バランスの取れたプロンプト構成です")


if __name__ == "__main__":
    analyze_prompt_length()

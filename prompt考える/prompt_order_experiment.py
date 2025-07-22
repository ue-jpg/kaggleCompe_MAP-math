"""
プロンプト内文章順序の実験
異なる順序でのプロンプトを比較
"""

import pandas as pd


def get_actual_labels_from_data():
    """実際のデータから全ラベルを取得"""
    try:
        df = pd.read_csv("processed_train_instructional.csv")
        return sorted(df["label"].unique())
    except FileNotFoundError:
        df = pd.read_csv("../map_data/train.csv")
        labels = (
            df["Category"].astype(str)
            + ":"
            + df["Misconception"].fillna("NA").astype(str)
        )
        return sorted(labels.unique())


def get_original_order_prompt(question, answer, explanation):
    """現在の順序（ベースライン）"""
    all_labels = get_actual_labels_from_data()
    labels_text = "\n".join([f"- {label}" for label in all_labels])

    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

TASK: Classify this student's response using EXACTLY ONE of these {len(all_labels)} labels:

{labels_text}

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons  
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

Classification:"""
    return prompt


def get_guidelines_first_prompt(question, answer, explanation):
    """ガイドライン優先順序"""
    all_labels = get_actual_labels_from_data()
    labels_text = "\n".join([f"- {label}" for label in all_labels])

    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons  
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

TASK: Classify this student's response using EXACTLY ONE of these {len(all_labels)} labels:

{labels_text}

Classification:"""
    return prompt


def get_compact_labels_prompt(question, answer, explanation):
    """ラベルを簡潔にまとめた版"""
    all_labels = get_actual_labels_from_data()

    # ラベルをカテゴリ別に整理
    true_correct = [l for l in all_labels if l.startswith("True_Correct")]
    false_correct = [l for l in all_labels if l.startswith("False_Correct")]
    true_neither = [l for l in all_labels if l.startswith("True_Neither")]
    false_neither = [l for l in all_labels if l.startswith("False_Neither")]
    true_misconceptions = [l for l in all_labels if l.startswith("True_Misconception")]
    false_misconceptions = [
        l for l in all_labels if l.startswith("False_Misconception")
    ]

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

AVAILABLE LABELS ({len(all_labels)} total):

Basic Categories:
{chr(10).join([f"- {l}" for l in true_correct + false_correct + true_neither + false_neither])}

True_Misconception Types ({len(true_misconceptions)}):
{chr(10).join([f"- {l}" for l in true_misconceptions])}

False_Misconception Types ({len(false_misconceptions)}):
{chr(10).join([f"- {l}" for l in false_misconceptions])}

Classification:"""
    return prompt


def get_context_focus_prompt(question, answer, explanation):
    """コンテキスト重視版（問題情報を強調）"""
    all_labels = get_actual_labels_from_data()
    labels_text = "\n".join([f"- {label}" for label in all_labels])

    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

=== STUDENT RESPONSE ANALYSIS ===
Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

=== ANALYSIS TASK ===
Carefully examine the student's explanation above. Determine if:
1. The student's reasoning is mathematically correct
2. The student demonstrates any specific misconceptions
3. The final answer aligns with correct mathematical thinking

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons  
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

Select EXACTLY ONE classification from these {len(all_labels)} options:
{labels_text}

Classification:"""
    return prompt


def compare_prompt_orders():
    """異なる順序のプロンプトを比較"""

    # サンプルデータ
    sample_question = "What is 1/2 + 1/3?"
    sample_answer = "5/6"
    sample_explanation = "I added 1+1=2 and 2+3=5, so the answer is 2/5"

    print("=== プロンプト順序比較実験 ===\n")

    # 1. オリジナル順序
    original = get_original_order_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print("1. オリジナル順序 (現在):")
    print(f"   長さ: {len(original)} 文字")
    print(f"   構造: 役割→問題→タスク→ラベル→ガイドライン→回答")
    print()

    # 2. ガイドライン優先
    guidelines_first = get_guidelines_first_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print("2. ガイドライン優先順序:")
    print(f"   長さ: {len(guidelines_first)} 文字")
    print(f"   構造: 役割→ガイドライン→問題→タスク→ラベル→回答")
    print()

    # 3. コンパクトラベル
    compact = get_compact_labels_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print("3. ラベル整理版:")
    print(f"   長さ: {len(compact)} 文字")
    print(f"   構造: 役割→問題→ガイドライン→整理済みラベル→回答")
    print()

    # 4. コンテキスト重視
    context_focus = get_context_focus_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print("4. コンテキスト重視版:")
    print(f"   長さ: {len(context_focus)} 文字")
    print(f"   構造: 役割→強調された問題→分析指示→ガイドライン→ラベル→回答")
    print()

    # 推奨順序の説明
    print("=== 推奨される改善点 ===")
    print("1. 重要な分類ガイドラインを早い位置に配置")
    print("2. ラベルリストをカテゴリ別に整理して可読性向上")
    print("3. 問題コンテキストを強調してLLMの注意を集中")
    print("4. タスク指示を明確で簡潔に")

    return {
        "original": original,
        "guidelines_first": guidelines_first,
        "compact": compact,
        "context_focus": context_focus,
    }


if __name__ == "__main__":
    prompts = compare_prompt_orders()

    print("\n=== サンプルプロンプト詳細 ===")
    print("\nコンテキスト重視版（推奨）:")
    print(prompts["context_focus"])

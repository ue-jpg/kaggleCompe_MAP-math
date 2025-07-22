"""
改良版プロンプトテンプレート - 全ラベル付き
instructionalプロンプトに全ての可能なラベルを追加
"""


def get_all_possible_labels():
    """データセットに存在する全ラベルのリスト"""
    return [
        "True_Correct:NA",
        "False_Neither:NA",
        "True_Neither:NA",
        "False_Misconception:Incomplete",
        "False_Misconception:Additive",
        "False_Misconception:Duplication",
        "False_Misconception:Subtraction",
        "False_Misconception:Positive",
        "False_Misconception:Wrong_term",
        "False_Misconception:Wrong_fraction",
        "False_Misconception:Ordering",
        "False_Misconception:Magnification",
        "False_Misconception:Different_variable",
        "False_Misconception:Multiplication",
        "False_Misconception:Rounding",
        "False_Misconception:Division",
        "False_Misconception:Conversion",
        "False_Misconception:Method",
        "False_Misconception:Operation",
        "False_Misconception:Algebra",
        "False_Misconception:Variable",
        "False_Misconception:Estimation",
        "False_Misconception:Negative",
        "False_Misconception:Comparing",
        "False_Misconception:Grouping",
        "False_Misconception:Form",
        "False_Misconception:Translation",
        "False_Misconception:Wrong_direction",
        "False_Misconception:Notation",
        "False_Misconception:Time",
        "False_Misconception:Perimeter",
        "False_Misconception:Power",
        "False_Misconception:Units",
        "False_Misconception:Magnitude",
        "True_Misconception:Incomplete",
        "True_Misconception:Wrong_term",
        "True_Misconception:Additive",
        "True_Misconception:Duplication",
        "True_Misconception:Subtraction",
        "True_Misconception:Wrong_fraction",
        "True_Misconception:Positive",
        "True_Misconception:Ordering",
        "True_Misconception:Different_variable",
        "True_Misconception:Multiplication",
        "True_Misconception:Magnification",
        "True_Misconception:Rounding",
        "True_Misconception:Division",
        "True_Misconception:Conversion",
        "True_Misconception:Method",
        "True_Misconception:Operation",
        "True_Misconception:Algebra",
        "True_Misconception:Variable",
        "True_Misconception:Estimation",
        "True_Misconception:Negative",
        "True_Misconception:Comparing",
        "True_Misconception:Grouping",
        "True_Misconception:Form",
        "True_Misconception:Translation",
        "True_Misconception:Wrong_direction",
        "True_Misconception:Notation",
        "True_Misconception:Time",
        "True_Misconception:Perimeter",
        "True_Misconception:Power",
        "True_Misconception:Units",
        "True_Misconception:Magnitude",
        "True_Misconception:Not_variable",
        "True_Misconception:Whole_numbers_larger",
        "True_Misconception:Adding_across",
        "True_Misconception:Longer_is_bigger",
        "True_Misconception:Base_rate",
    ]


def get_enhanced_instructional_prompt(question, answer, explanation):
    """全ラベル付きの改良版instructionalプロンプト"""

    # 主要カテゴリーの説明
    main_categories = """
MAIN CATEGORIES:
- True_Correct: Student demonstrates correct understanding (no misconception)
- True_Neither: Correct answer but unclear/incomplete reasoning (no specific misconception) 
- False_Neither: Incorrect answer but no identifiable misconception
- False_Misconception: Incorrect answer with specific mathematical misconception
- True_Misconception: Correct answer but shows specific mathematical misconception"""

    # 具体的な誤解の種類
    misconception_types = """
SPECIFIC MISCONCEPTIONS (if applicable):
Mathematical Operations:
- Additive: Incorrectly adding instead of other operations
- Subtraction: Incorrect subtraction methods or concepts
- Multiplication: Multiplication errors or misconceptions
- Division: Division errors or misconceptions
- Duplication: Incorrectly duplicating values or operations

Number Concepts:
- Positive: Misconceptions about positive numbers
- Negative: Misconceptions about negative numbers  
- Rounding: Incorrect rounding procedures
- Ordering: Mistakes in ordering numbers
- Magnitude: Misconceptions about number size/magnitude
- Whole_numbers_larger: Believing whole numbers are always larger

Fractions & Decimals:
- Wrong_fraction: Incorrect fraction concepts or operations
- Conversion: Errors in converting between forms

Algebra & Variables:
- Algebra: General algebraic misconceptions
- Variable: Misconceptions about variables
- Different_variable: Treating different variables incorrectly
- Not_variable: Not recognizing variables

Geometry & Measurement:
- Perimeter: Incorrect perimeter calculations or concepts
- Time: Time-related misconceptions
- Units: Unit conversion or usage errors
- Magnification: Scale or magnification errors
- Longer_is_bigger: Assuming longer means bigger

Problem-Solving Methods:
- Method: Using incorrect mathematical methods
- Operation: Choosing wrong operations
- Estimation: Poor estimation strategies
- Incomplete: Incomplete reasoning or solutions
- Wrong_term: Using incorrect mathematical terminology

Representation & Notation:
- Form: Misconceptions about mathematical forms
- Notation: Incorrect mathematical notation usage
- Translation: Errors in translating between representations
- Wrong_direction: Directional errors in operations or concepts

Comparison & Logic:
- Comparing: Errors in comparing quantities
- Grouping: Incorrect grouping strategies
- Power: Misconceptions about powers/exponents
- Base_rate: Base rate fallacy or related errors
- Adding_across: Incorrectly adding across different contexts"""

    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

{main_categories}

{misconception_types}

CLASSIFICATION TASK:
Analyze this student's mathematical reasoning and classify it using EXACTLY ONE of the following formats:
- Category:NA (for True_Correct, True_Neither, or False_Neither)
- Category:SpecificMisconception (for any misconception type listed above)

Examples:
- True_Correct:NA
- False_Misconception:Additive  
- True_Misconception:Incomplete
- False_Neither:NA

Provide your classification in the format: Category:Misconception"""

    return prompt


def get_compact_enhanced_prompt(question, answer, explanation):
    """よりコンパクトな全ラベル付きプロンプト"""

    all_labels = get_all_possible_labels()
    labels_text = "\n".join([f"- {label}" for label in all_labels])

    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

TASK: Classify this student's response using EXACTLY ONE of these labels:

{labels_text}

GUIDELINES:
- True_Correct:NA = Demonstrates correct understanding
- True_Neither:NA = Correct answer, unclear reasoning
- False_Neither:NA = Incorrect answer, no specific misconception
- True_Misconception:[Type] = Correct answer but shows misconception
- False_Misconception:[Type] = Incorrect answer with specific misconception

Classification:"""

    return prompt


# テスト用の関数
def test_enhanced_prompts():
    """プロンプトの例を表示"""
    sample_question = "What is 2/3 + 1/4?"
    sample_answer = "11/12"
    sample_explanation = "I added 2+1=3 and 3+4=7, so the answer is 3/7"

    print("=== 詳細版プロンプト ===")
    detailed = get_enhanced_instructional_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print(detailed[:500] + "...\n")

    print("=== コンパクト版プロンプト ===")
    compact = get_compact_enhanced_prompt(
        sample_question, sample_answer, sample_explanation
    )
    print(compact[:500] + "...\n")


if __name__ == "__main__":
    test_enhanced_prompts()

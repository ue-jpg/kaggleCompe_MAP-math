"""
プロンプトテンプレートの定義
異なるプロンプト戦略を試すためのテンプレート集
"""


def get_basic_prompt(question, answer, explanation):
    """基本的なプロンプト（現在使用中）"""
    return f"""Question: {question}
Correct Answer: {answer}
Student Explanation: {explanation}"""


def get_instructional_prompt(question, answer, explanation):
    """指示明確型プロンプト"""
    return f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Correct Answer: {answer}
Student's Explanation: {explanation}

Task: Analyze this student's mathematical reasoning and classify it as:
- True_Correct: Student demonstrates correct understanding
- True_Neither: Correct answer but unclear/incomplete reasoning  
- False_Neither: Incorrect answer but no identifiable misconception
- False_Misconception: Demonstrates a specific mathematical misconception

Provide your classification in the format: Category:Misconception"""


def get_step_by_step_prompt(question, answer, explanation):
    """思考プロセス誘導型プロンプト"""
    return f"""As a mathematics teacher, evaluate this student's reasoning systematically.

Mathematical Problem: {question}
Expected Answer: {answer}
Student's Response: "{explanation}"

Evaluation Process:
1. Analyze the mathematical accuracy of the student's reasoning
2. Identify any conceptual errors or misconceptions
3. Determine if the explanation demonstrates true understanding

Based on your analysis, classify this response using the format: Category:Misconception"""


def get_few_shot_prompt(question, answer, explanation):
    """例示付きプロンプト"""
    return f"""Classify student mathematical explanations for misconception analysis.

Classification Examples:
- Student: "I just added 2+3=5 and 4+5=9, so 2/4 + 3/5 = 5/9"
  Classification: False_Misconception:Additive

- Student: "Since 3×4=12 and 12÷4=3, the answer is correct"
  Classification: True_Correct:NA

- Student: "I think it's 7 because that feels right"
  Classification: False_Neither:NA

Now classify this response:
Question: {question}
Correct Answer: {answer}
Student Explanation: {explanation}

Classification:"""


def get_structured_prompt(question, answer, explanation):
    """構造化分析型プロンプト"""
    return f"""MATHEMATICAL MISCONCEPTION ANALYSIS

PROBLEM STATEMENT: {question}
CORRECT SOLUTION: {answer}
STUDENT RESPONSE: {explanation}

ANALYSIS FRAMEWORK:
□ Mathematical Accuracy: Is the reasoning mathematically sound?
□ Conceptual Understanding: Does the student grasp the underlying concept?
□ Error Identification: What specific misconceptions (if any) are present?

CLASSIFICATION REQUIRED: Provide in format Category:Misconception

Your Analysis:"""


def get_contextual_prompt(question, answer, explanation):
    """文脈重視型プロンプト"""
    return f"""You are evaluating student understanding in mathematics education.

Context: This is a formative assessment to identify specific learning gaps and misconceptions that require targeted intervention.

Student Work:
Problem: {question}
Expected Answer: {answer}
Student's Explanation: {explanation}

Your task: Determine whether this student response indicates:
1. Solid conceptual understanding (True_Correct)
2. Correct answer with weak reasoning (True_Neither) 
3. Incorrect answer without clear misconception (False_Neither)
4. Specific identifiable misconception (False_Misconception)

Provide classification as Category:Misconception format."""


# プロンプトテンプレートの辞書
PROMPT_TEMPLATES = {
    "basic": get_basic_prompt,
    "instructional": get_instructional_prompt,
    "step_by_step": get_step_by_step_prompt,
    "few_shot": get_few_shot_prompt,
    "structured": get_structured_prompt,
    "contextual": get_contextual_prompt,
}


def get_prompt(template_name, question, answer, explanation):
    """指定されたテンプレートでプロンプトを生成"""
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name](question, answer, explanation)
    else:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(PROMPT_TEMPLATES.keys())}"
        )

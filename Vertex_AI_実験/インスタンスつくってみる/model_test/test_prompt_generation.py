import pandas as pd

DATA_PATH = r"C:\Users\mouse\Desktop\NotDelete\GitHub\kaggleCompe_MAP-math\map_data\train.csv"

# データ読み込み
train_df = pd.read_csv(DATA_PATH)
all_labels = (
    train_df["Category"].astype(str) + ":" + train_df["Misconception"].fillna("NA").astype(str)
).unique()
all_labels = sorted(all_labels)

def get_improved_compact_prompt(question, answer, explanation, all_labels):
    labels_text = "\n".join([f"- {label}" for label in all_labels])
    prompt = f"""You are an expert math educator analyzing student responses for mathematical misconceptions.

Question: {question}
Student's Answer: {answer}
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

def create_enhanced_text_with_improved_prompt(row, all_labels):
    question = str(row["QuestionText"]) if pd.notna(row["QuestionText"]) else ""
    mc_answer = str(row["MC_Answer"]) if pd.notna(row["MC_Answer"]) else ""
    explanation = str(row["StudentExplanation"]) if pd.notna(row["StudentExplanation"]) else ""
    return get_improved_compact_prompt(question, mc_answer, explanation, all_labels)

# 先頭5件でテスト
for i, row in train_df.head(5).iterrows():
    print(f"\n--- Sample {i+1} ---")
    prompt = create_enhanced_text_with_improved_prompt(row, all_labels)
    print(prompt)
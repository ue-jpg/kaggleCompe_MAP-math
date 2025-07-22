import pandas as pd

# データ読み込み
df = pd.read_csv("../map_data/train.csv")
labels = df["Category"].astype(str) + ":" + df["Misconception"].fillna("NA").astype(str)
all_labels = sorted(labels.unique())

print("全ラベル一覧 (アルファベット順):")
for i, label in enumerate(all_labels, 1):
    print(f"{i:2d}. {label}")

print(f"\n総ラベル数: {len(all_labels)}")

# False_Correctがあるかチェック
false_correct_labels = [
    label for label in all_labels if label.startswith("False_Correct")
]
print(f"\nFalse_Correctラベル: {false_correct_labels}")

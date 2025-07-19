#!/usr/bin/env python3
"""
MAP - Charting Student Math Misunderstandings コンペ専用ダウンローダー

このスクリプトは特定のコンペのデータを直接ダウンロードします。
"""

import os
import pandas as pd


def check_kaggle_api():
    """Kaggle APIが利用可能かチェック"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # APIキーファイルの存在確認
        kaggle_json_path = os.path.join(
            os.path.expanduser("~"), ".kaggle", "kaggle.json"
        )
        if not os.path.exists(kaggle_json_path):
            return False, "APIキーファイルが見つかりません"
        return True, None
    except ImportError:
        return False, "Kaggle APIがインストールされていません"


def download_map_competition():
    """MAP - Charting Student Math Misunderstandings のデータをダウンロード"""

    competition_name = "map-charting-student-math-misunderstandings"
    download_path = "./map_data"

    # APIキーの確認
    api_available, error_msg = check_kaggle_api()
    if not api_available:
        print(f"❌ Kaggle API エラー: {error_msg}")
        print("\n設定が必要な項目:")
        print("1. Kaggle.com にログインして https://www.kaggle.com/account に移動")
        print("2. 'API' セクションで 'Create New API Token' をクリック")
        print("3. ダウンロードした kaggle.json を C:\\Users\\mouse\\.kaggle\\ に配置")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Kaggle API をセットアップ
        api = KaggleApi()
        api.authenticate()

        # ダウンロードディレクトリを作成
        os.makedirs(download_path, exist_ok=True)

        print(f"'{competition_name}' のデータをダウンロード中...")

        # コンペのデータをダウンロード（zipファイルをダウンロードして後で展開）
        api.competition_download_files(competition_name, path=download_path)

        # zipファイルを展開
        import zipfile

        zip_files = [f for f in os.listdir(download_path) if f.endswith(".zip")]
        for zip_file in zip_files:
            zip_path = os.path.join(download_path, zip_file)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(download_path)
            # zipファイルを削除
            os.remove(zip_path)

        print(f"ダウンロード完了: {download_path}")

        # ダウンロードしたファイルを一覧表示
        files = os.listdir(download_path)
        print("\nダウンロードしたファイル:")
        for file in sorted(files):
            file_path = os.path.join(download_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("\n設定が必要な項目:")
        print("1. Kaggle.com にログインして https://www.kaggle.com/account に移動")
        print("2. 'API' セクションで 'Create New API Token' をクリック")
        print("3. ダウンロードした kaggle.json を C:\\Users\\mouse\\.kaggle\\ に配置")
        return False


def load_map_data(data_path="./map_data"):
    """MAPコンペのデータを読み込み"""

    data_dict = {}

    if not os.path.exists(data_path):
        print(f"データパス '{data_path}' が見つかりません。")
        print("まず download_map_competition() を実行してください。")
        return data_dict

    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    print("データファイルを読み込み中...")
    for csv_file in csv_files:
        file_path = os.path.join(data_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            data_dict[csv_file.replace(".csv", "")] = df
            print(f"✓ {csv_file}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            print(f"✗ {csv_file} の読み込みに失敗: {e}")

    return data_dict


def explore_map_data(data_dict):
    """MAPコンペデータの詳細探索"""

    if not data_dict:
        print("データが読み込まれていません。")
        return

    for name, df in data_dict.items():
        print(f"\n{'='*60}")
        print(f"データセット: {name}")
        print(f"{'='*60}")
        print(f"形状: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"メモリ使用量: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"\n列の詳細:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = null_count / len(df) * 100
            print(
                f"  {i:2d}. {col:<30} | {str(dtype):<10} | 欠損値: {null_count:>6} ({null_pct:5.1f}%)"
            )

        print(f"\n最初の3行:")
        print(df.head(3).to_string())

        # 数値列の統計
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(f"\n数値列の基本統計:")
            print(df[numeric_cols].describe().round(2))


if __name__ == "__main__":
    print("MAP - Charting Student Math Misunderstandings コンペ")
    print("=" * 50)

    # データをダウンロード
    print("1. データをダウンロードしています...")
    if download_map_competition():
        print("\n2. データを読み込んでいます...")
        # データを読み込み
        data = load_map_data()

        if data:
            print("\n3. データを探索しています...")
            # データを探索
            explore_map_data(data)

            print(f"\n✓ 完了! データは './map_data' ディレクトリに保存されました。")
            print("データフレームは以下の変数でアクセスできます:")
            for name in data.keys():
                print(f"  - {name}")
        else:
            print("データの読み込みに失敗しました。")
    else:
        print("データのダウンロードに失敗しました。")

# data_prep.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT_CSV = "NetFlix.csv"
OUTPUT_CSV = "prepared_data.csv"

def main():
    # 1. Загрузка исходных данных
    df = pd.read_csv(INPUT_CSV)

    # 2. Целевая переменная: Movie (1) / TV Show (0)
    df = df.copy()
    df = df[df["type"].notna()]
    df["target_is_movie"] = (df["type"] == "Movie").astype(int)

    # 3. Оставляем осмысленные признаки
    cols_keep = [
        "release_year",
        "rating",
        "duration",
        "country",
        "genres",
    ]
    df = df[cols_keep + ["target_is_movie"]]

    # 4. Пропуски
    cat_cols = ["rating", "country", "genres"]
    num_cols = ["release_year", "duration"]

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    # duration: строки вида "90", "4", "90 min", "4 Seasons" и т.п.
    # Вытащим первое число, остальное отбросим.
    def parse_duration(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        # берем первое число в строке
        digits = ""
        for ch in s:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        return float(digits) if digits else np.nan

    df["duration_num"] = df["duration"].apply(parse_duration)
    df["duration_num"] = df["duration_num"].fillna(df["duration_num"].median())
    df.drop(columns=["duration"], inplace=True)
    num_cols = ["release_year", "duration_num"]

    # 5. Разделение на X и y
    X_cat = df[cat_cols]
    X_num = df[num_cols]
    y = df["target_is_movie"]

    # 6. One-Hot кодирование категориальных
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_enc = ohe.fit_transform(X_cat)

    # 7. Масштабирование числовых
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # 8. Объединение
    X = np.hstack([X_cat_enc, X_num_scaled])
    feature_names = list(ohe.get_feature_names_out(cat_cols)) + num_cols

    df_prepared = pd.DataFrame(X, columns=feature_names)
    df_prepared["target_is_movie"] = y.values

    # 9. Сохранение
    df_prepared.to_csv(OUTPUT_CSV, index=False)
    print(f"Prepared data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

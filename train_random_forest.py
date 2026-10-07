import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file = "city_day.csv"

df = pd.read_csv(file)

print("========================================")
print("       AIR QUALITY DATASET")
print("========================================")

print("Total records:", len(df))
print("Total cities :", df["City"].nunique())


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "O3"
]

target = "AQI_Bucket"


# ============================================================
# 3. KEEP ONLY REQUIRED COLUMNS
# ============================================================

df = df[features + [target]].copy()


# ============================================================
# 4. REMOVE ROWS WHERE TARGET IS MISSING
# ============================================================

df = df.dropna(subset=[target])

print("\nRecords with AQI_Bucket:", len(df))


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n----------------------------------------")
print("Missing Values")
print("----------------------------------------")

print(df[features].isnull().sum())


# ============================================================
# 6. INPUT AND TARGET
# ============================================================

X = df[features]

y = df[target]


# ============================================================
# 7. HANDLE MISSING POLLUTANT VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features,
    index=X.index
)


print("\nMissing values after imputation:")

print(X.isnull().sum())


# ============================================================
# 8. DISPLAY AIR QUALITY CLASSES
# ============================================================

print("\n----------------------------------------")
print("Air Quality Categories")
print("----------------------------------------")

print(y.value_counts())


# ============================================================
# 9. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n----------------------------------------")
print("Dataset Split")
print("----------------------------------------")

print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 10. CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n----------------------------------------")
print("Training Random Forest...")
print("----------------------------------------")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 12. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("       RANDOM FOREST RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

print("\n----------------------------------------")
print("Feature Importance")
print("----------------------------------------")

for feature, importance in zip(
    features,
    model.feature_importances_
):

    print(
        f"{feature:10s} : {importance:.4f}"
    )


# ============================================================
# 17. SAVE RANDOM FOREST MODEL
# ============================================================

joblib.dump(
    model,
    "air_quality_rf_model.pkl"
)


# ============================================================
# 18. SAVE IMPUTER
# ============================================================

joblib.dump(
    imputer,
    "air_quality_imputer.pkl"
)


print("\n========================================")
print("       MODEL SAVED SUCCESSFULLY")
print("========================================")

print("Random Forest : air_quality_rf_model.pkl")
print("Imputer       : air_quality_imputer.pkl")


# ============================================================
# 19. TEST WITH YOUR LIVE OPEN-METEO DATA
# ============================================================

live_data = pd.DataFrame(
    [[
        15.3,     # PM2.5
        17.2,     # PM10
        21.5,     # NO2
        5.1,      # SO2
        19.0      # O3
    ]],
    columns=features
)


# Apply same imputer
live_data_imputed = imputer.transform(
    live_data
)


# Predict
prediction = model.predict(
    live_data_imputed
)


print("\n========================================")
print("       LIVE DATA TEST")
print("========================================")

print("PM2.5 :", live_data["PM2.5"].iloc[0])
print("PM10  :", live_data["PM10"].iloc[0])
print("NO2   :", live_data["NO2"].iloc[0])
print("SO2   :", live_data["SO2"].iloc[0])
print("O3    :", live_data["O3"].iloc[0])

print("----------------------------------------")

print(
    "Predicted Air Quality:",
    prediction[0]
)

print("========================================")
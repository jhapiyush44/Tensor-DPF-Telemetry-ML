import joblib

reg = joblib.load("models/regressor.pkl")
clf = joblib.load("models/classifier.pkl")

if hasattr(reg, "feature_names_in_"):
    print("REGRESSOR FEATURES:", list(reg.feature_names_in_))
else:
    print("REGRESSOR feature count:", len(reg.feature_importances_))

if hasattr(clf, "feature_names_in_"):
    print("CLASSIFIER FEATURES:", list(clf.feature_names_in_))
else:
    print("CLASSIFIER feature count:", len(clf.feature_importances_))
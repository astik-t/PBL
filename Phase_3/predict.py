import pickle, json, numpy as np
from collections import Counter

MODELS = {
    '1': ('Logistic Regression', 'model_lr.pkl'),
    '2': ('SVM Linear',          'model_svm_linear.pkl'),
    '3': ('SVM Polynomial',      'model_svm_poly.pkl'),
    '4': ('SVM RBF',             'model_svm_rbf.pkl'),
    '5': ('Random Forest',       'model_rf.pkl'),
}
LABEL_MAP = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}

with open('selected_features.json') as f:
    FEATURES = json.load(f)

print("\nSelect model:")
for k, (name, _) in MODELS.items():
    print(f"  {k}. {name}")
print("  6. All (majority vote)")
choice = input("\nEnter choice (1-6): ").strip()

print(f"\nFeatures: {FEATURES}")
raw = input("Enter values as comma-separated list: ").split(',')
input_arr = np.array([float(v.strip()) for v in raw]).reshape(1, -1)

print("\n" + "="*45)
if choice == '6':
    preds = {}
    for name, path in MODELS.values():
        with open(path, 'rb') as f:
            model = pickle.load(f)
        p = LABEL_MAP[model.predict(input_arr)[0]]
        preds[name] = p
        print(f"  {name:<22} → {p}")
    vote = Counter(preds.values()).most_common(1)[0][0]
    print(f"\n  Majority Vote → {vote}")
else:
    name, path = MODELS[choice]
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"  {name:<22} → {LABEL_MAP[model.predict(input_arr)[0]]}")
print("="*45)
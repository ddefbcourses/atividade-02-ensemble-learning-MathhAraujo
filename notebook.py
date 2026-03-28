import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data(seed=42):
    mnist_fashion = fetch_openml('Fashion-MNIST', version=1, return_X_y=False, as_frame=False)
    X = mnist_fashion.data
    y = mnist_fashion.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data(seed=42)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def train_random_forest(X_train, y_train, seed=42):
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model

def train_adaboost(X_train, y_train, seed=42):
    model = AdaBoostClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    return model

random_forest = train_random_forest(X_train, y_train)
adaboost = train_adaboost(X_train, y_train)

from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

rf_train_pred = random_forest.predict(X_train)
rf_test_pred = random_forest.predict(X_test)
ab_train_pred = adaboost.predict(X_train)
ab_test_pred = adaboost.predict(X_test)

print("Random Forest:")
print(f"  Acuracia treino: {accuracy_score(y_train, rf_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, rf_test_pred)}")

print("\nAdaBoost:")
print(f"  Acuracia treino: {accuracy_score(y_train, ab_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, ab_test_pred)}")

def run_pipeline(model_type="rf", seed=42):
    X_train, X_test, y_train, y_test = load_data(seed=seed)
    if model_type == "rf":
        model = train_random_forest(X_train, y_train, seed=seed)
    elif model_type == "ab":
        model = train_adaboost(X_train, y_train, seed=seed)
    return evaluate(model, X_test, y_test)

from sklearn.tree import DecisionTreeClassifier

depths = list(range(5, 51, 5))

print("=== Random Forest ===")
print(f"{'Profundidade':<15} {'Acc Treino':<15} {'Acc Teste':<15}")
prev_acc_test = 0
for d in depths:
    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
    rf.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, rf.predict(X_train))
    acc_test = accuracy_score(y_test, rf.predict(X_test))
    print(f"{d:<15} {acc_train:<15.4f} {acc_test:<15.4f}")
    if acc_test < prev_acc_test:
        print(f"Overfitting detectado na profundidade {d} (acc teste caiu de {prev_acc_test:.4f} para {acc_test:.4f})")
        break
    prev_acc_test = acc_test

print("\n=== AdaBoost ===")
print(f"{'Profundidade':<15} {'Acc Treino':<15} {'Acc Teste':<15}")
prev_acc_test = 0
for d in depths:
    ab = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=d),
        n_estimators=100, random_state=42
    )
    ab.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, ab.predict(X_train))
    acc_test = accuracy_score(y_test, ab.predict(X_test))
    print(f"{d:<15} {acc_train:<15.4f} {acc_test:<15.4f}")
    if acc_test < prev_acc_test:
        print(f"Overfitting detectado na profundidade {d} (acc teste caiu de {prev_acc_test:.4f} para {acc_test:.4f})")
        break
    prev_acc_test = acc_test

from sklearn.metrics import classification_report

X_tr, X_te, y_tr, y_te = load_data(seed=42)

rf = train_random_forest(X_tr, y_tr)
print("Random Forest:")
print(classification_report(y_te, rf.predict(X_te)))

print("\n=========================================================\n")

ab = train_adaboost(X_tr, y_tr)
print("AdaBoost:")
print(classification_report(y_te, ab.predict(X_te)))

print("Random Forest (seed=42):")
print(f"  Acuracia: {run_pipeline('rf', 42):.4f}")
print("\nRandom Forest (seed=7):")
print(f"  Acuracia: {run_pipeline('rf', 7):.4f}")

print("\n=========================================================\n")

print("AdaBoost (seed=42):")
print(f"  Acuracia: {run_pipeline('ab', 42):.4f}")
print("\nAdaBoost (seed=7):")
print(f"  Acuracia: {run_pipeline('ab', 7):.4f}")

print("Random Forest:")
print(f"  Acuracia treino: {accuracy_score(y_train, rf_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, rf_test_pred)}")

print("\nAdaBoost:")
print(f"  Acuracia treino: {accuracy_score(y_train, ab_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, ab_test_pred)}")

print("Random Forest sem variaÃ§Ã£o de hiperparÃ¢metro:")
print(classification_report(y_test, rf_test_pred))

print(f"  Acuracia treino: {accuracy_score(y_train, rf_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, rf_test_pred)}")

print("\n=========================================================\n")

#Por padrÃ£o utiliza max_depth sem limites, passando a base pode-se limitar a profundidade.
random_forest_v2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
random_forest_v2.fit(X_train, y_train)
rf_v2_pred = random_forest_v2.predict(X_test)

rf_v2_train_pred = random_forest_v2.predict(X_train)

print("Random Forest com variaÃ§Ã£o de hiperparÃ¢metro:")
print(classification_report(y_test, rf_v2_pred))

print(f"  Acuracia treino: {accuracy_score(y_train, rf_v2_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, rf_v2_pred)}")

print("\n=========================================================\n")

print("Adaboost sem variaÃ§Ã£o de hiperparÃ¢metro:")
print(classification_report(y_test, ab_test_pred))
print(f"  Acuracia treino: {accuracy_score(y_train, ab_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, ab_test_pred)}")

print("\n=========================================================\n")

#Aqui o problema Ã© contrÃ¡rio, por padrÃ£o o max_depth Ã© 1.
adaboost_v2 = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=42)
adaboost_v2.fit(X_train, y_train)
ab_v2_pred = adaboost_v2.predict(X_test)

ab_v2_train_pred = adaboost_v2.predict(X_train)

print("Adaboost com variaÃ§Ã£o de hiperparÃ¢metro:")
print(classification_report(y_test, ab_v2_pred))

print(f"  Acuracia treino: {accuracy_score(y_train, ab_v2_train_pred)}")
print(f"  Acuracia teste:  {accuracy_score(y_test, ab_v2_pred)}")




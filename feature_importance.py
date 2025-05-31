import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_feature_importance(
    model_predict,
    transform_fn,
    X,
    y,
    metric_fn,
    n_repeats=5,
    n_jobs=1,
    random_state=None
):
    """
    Calcule l'importance des features pour un modèle de régression à l'aide de l'approche de permutation,
    en tenant compte d'une fonction de transformation avec des règles métiers.

    Paramètres
    ----------
    model_predict : function
        Fonction qui prend en entrée un tableau 2D (features transformées) et renvoie les prédictions du modèle.
    transform_fn : function
        Fonction de transformation qui applique les règles métiers à l'ensemble de données d'entrée X brut.
        Doit retourner un tableau 2D (ou DataFrame) de features transformées.
    X : pandas.DataFrame ou numpy.ndarray
        Jeu de données brut (avant transformation). Dimensions : (n_samples, n_features).
    y : array-like
        Vecteur des cibles réelles. Dimensions : (n_samples,).
    metric_fn : function
        Fonction mesurant la performance du modèle, par exemple mean_squared_error.
        Doit prendre en entrée (y_true, y_pred) et retourner un score (float). Ici, on suppose
        que plus le score est bas, meilleur est le modèle (ex : MSE). L'importance sera la
        différence entre le score sur les données permutées et le score de référence.
    n_repeats : int, optionnel (défaut=5)
        Nombre de permutations à réaliser par feature pour estimer la distribution des scores.
    n_jobs : int, optionnel (défaut=1)
        Nombre de threads à utiliser pour le calcul en parallèle. Si ==1, pas de parallélisation.
    random_state : int ou None, optionnel
        Graine pour la reproductibilité des permutations.

    Retour
    ------
    importances : pandas.Series
        Série pandas contenant l'importance moyenne de chaque feature, indexée par le nom ou l'indice
        de la feature dans X. L'importance correspond à l'augmentation moyenne du score (metric_fn)
        lorsqu'on permute cette feature.
    """

    # 1. Fixer la graine si fournie
    rng = np.random.RandomState(random_state)

    # 2. Vérifier et convertir X en DataFrame pour conserver les noms de colonnes
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    elif isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        raise ValueError("X doit être un pandas.DataFrame ou un numpy.ndarray.")

    # 3. Dimensions : nombre d'échantillons et de features
    n_samples, n_features = X_df.shape

    # 4. Calcul du score de référence (sans permutation)
    X_transformed_ref = transform_fn(X_df)
    y_pred_ref = model_predict(X_transformed_ref)
    score_ref = metric_fn(y, y_pred_ref)

    # 5. Fonction interne pour calculer l'importance d'une seule feature
    def _wrapper_feature(feature_name):
        """
        Pour une feature donnée :
        - on permute plusieurs fois la colonne correspondante,
        - on applique la transformation, on calcule la prédiction et la métrique,
        - enfin on retourne (feature_name, importance_moyenne).
        """
        scores_perm = []
        for _ in range(n_repeats):
            # Copier X et permuter la colonne feature_name
            X_permuted = X_df.copy()
            permuted_values = X_permuted[feature_name].values.copy()
            rng.shuffle(permuted_values)
            X_permuted[feature_name] = permuted_values

            # Appliquer la transformation métier puis prédire
            X_transformed = transform_fn(X_permuted)
            y_pred_perm = model_predict(X_transformed)

            # Calculer le score sur les données permutées
            score_perm = metric_fn(y, y_pred_perm)
            scores_perm.append(score_perm)

        # Calculer la différence moyenne entre scores permutés et score de référence
        mean_diff = np.mean(np.array(scores_perm) - score_ref)
        return feature_name, mean_diff

    feature_names = list(X_df.columns)
    importances_dict = {}

    # 6. Parallélisation (ou séquentiel si n_jobs=1)
    if n_jobs == 1:
        # Boucle séquentielle
        for fname in feature_names:
            _, imp = _wrapper_feature(fname)
            importances_dict[fname] = imp
    else:
        # Utilisation de ThreadPoolExecutor pour éviter les problèmes de pickling
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_feature = {executor.submit(_wrapper_feature, fname): fname for fname in feature_names}
            for future in as_completed(future_to_feature):
                fname = future_to_feature[future]
                try:
                    _, imp = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Erreur lors du calcul de l'importance pour {fname}: {exc}")
                importances_dict[fname] = imp

    # 7. Retourner un pandas.Series ordonné (décroissant)
    importances = pd.Series(importances_dict).sort_values(ascending=False)
    return importances


# -----------------------------
# Exemple d'utilisation ci-dessous
# -----------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Exemple fictif de données
    rng = np.random.RandomState(42)
    X_demo = pd.DataFrame({
        "a": rng.normal(size=100),
        "b": rng.uniform(size=100),
        "c": rng.randint(0, 2, size=100)
    })

    # Fonction de transformation métier (exemple simple)
    def transform_fn_example(df):
        df_new = df.copy()
        df_new["a_squared"] = df_new["a"] ** 2
        df_new["b_log"] = np.log1p(df_new["b"])
        # On ne garde ici que les features transformées
        return df_new[["a_squared", "b_log", "c"]]

    # Génération de la cible (y) selon une formule connue + bruit
    y_demo = (
        2 * X_demo["a"] ** 2
        + 3 * np.log1p(X_demo["b"])
        + 5 * X_demo["c"]
        + rng.normal(scale=0.1, size=100)
    )

    # Entraînement d'un modèle linéaire sur les features transformées
    X_trans_demo = transform_fn_example(X_demo)
    model = LinearRegression().fit(X_trans_demo, y_demo)

    def model_predict_example(X_transformed):
        return model.predict(X_transformed)

    # Calcul des importances (avec 10 répétitions et 2 threads)
    importances = compute_feature_importance(
        model_predict=model_predict_example,
        transform_fn=transform_fn_example,
        X=X_demo,
        y=y_demo,
        metric_fn=mean_squared_error,
        n_repeats=10,
        n_jobs=2,
        random_state=42
    )

    print("Importances des features (en ordre décroissant) :")
    print(importances)
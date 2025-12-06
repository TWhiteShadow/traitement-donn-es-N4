# Projet N4 — Analyse de cibles : import, nettoyage, exploration et storytelling

**Auteur :**  
**Date :** `YYYY-MM-DD` (généré automatiquement après exécution)

---

## 0. Résumé exécutif
_Résumé court (3–5 phrases) présentant les objectifs, la méthodologie, et la principale conclusion (groupe vulnérable identifié)._

---

## 1. Import & optimisation mémoire

- **Fichier source :** `result.csv`  
- **Nombre de lignes initiales :** `{{n_rows_raw}}`  
- **Mémoire avant optimisation :** `{{memory_raw_kb}} KB`  
- **Nombre de lignes après nettoyage initial :** `{{n_rows_cleaned}}`  
- **Mémoire après optimisation :** `{{memory_after_kb}} KB`  

> Voir `output_assets/report_assets/summary_stats.csv` pour les chiffres exacts.

**Justification des choix d'optimisation**
- Types numériques castés en `int8`/`int16` pour réduire la mémoire.
- Colonnes textuelles en `category` si cardinalité faible.
- Suppression conservatrice des lignes manquantes sur colonnes numériques essentielles (scores, âge).

---

## 2. Nettoyage et mise en forme

### Problèmes détectés
- Voir `output_assets/report_assets/problems_detected.csv`.

### Actions réalisées
- Trim + lowercase sur colonnes texte (`recommended_product`, `canal_recommande`/`support`).
- Normalisation de `campaign_success` en booléen.
- Conversion des scores et de l'âge en numériques, coercition des erreurs en `NaN`.
- Suppression des lignes où un score ou l'âge est manquant (justification : colonnes essentielles).

**Impact mémoire** : voir section 1.

---

## 3. Détection d'anomalies

**Méthodes utilisées**
- Z-score (|z| > 3) — détecte valeurs éloignées de la moyenne.
- IQR (hors [Q1 - 1.5·IQR, Q3 + 1.5·IQR]) — robuste aux asymétries.

**Fichiers produits**
- `output_assets/anomalies.csv` : lignes considérées anormales (au moins une colonne).
- `output_assets/report_assets/anomaly_summary.csv` : résumé par colonne.

**Observations**  
- (À remplir après exécution : nombre anomalies gaming, insta, football, âge)
- Graphiques : `output_assets/plots/boxplot_gaming_by_age.png`, `scatter_gaming_age_anomalies.png`, etc.

---

## 4. Analyse statistique (KPI)

KPI calculés :
- Taux de réussite global : `{{global_success_rate}}` (voir `kpi_summary.csv`)
- Taux de réussite par produit : `output_assets/report_assets/kpi_success_by_product.csv`
- Taux de réussite par support : `output_assets/report_assets/kpi_success_by_support.csv`
- Taux de réussite par tranche d'âge : `output_assets/report_assets/kpi_success_by_agegroup.csv`
- Taux par segment d'intérêt (seuil 75) : inclut `gaming_interest_score_high` etc.

Graphiques fournis :
- `output_assets/plots/bar_top_products.png`
- `output_assets/plots/bar_success_by_support.png`
- `output_assets/plots/bar_success_by_agegroup.png`
- Histograms & boxplots en `output_assets/plots/`

**Observations principales**
- (À remplir après exécution : tendances, corrélations fortes/weak, limites)

---

## 5. Datatelling — Exemple de scénario d'attaque (expliquer / justifier)
**Choix du groupe cible :** (ex : jeunes 18–24 avec gaming_interest_score >= 75)

**Données supportant ce choix :**
- (Chiffrer : n, taux de succès, score moyen, support préféré)

**Scénario d'attaque proposé :**
- Support : Instagram / TikTok / Email (selon KPI)
- Message : ex. "Récompense / Giveaway lié à jeu populaire"
- CTA : lien de phishing simulé

**Justification :**
- (Fournir corrélations et graphiques supportant le scénario)

---

## 6. Bonus — Modèle prédictif (résumé)
- Modèles entrainés : LogisticRegression, DecisionTree
- Fichier métriques : `output_assets/report_assets/ml_metrics.csv`
- Modèle sauvegardé (meilleur) : `output_assets/model_joblib.pkl`

**Utilité pédagogique** : montre comment on pourrait prédire la probabilité qu'un individu clique, afin d'orienter priorisation / prévention.

---

## 7. Limites & recommandations
- Données synthétiques — prudence pour généralisation.
- Prochaine étape : enrichir features (comportements temporels, device, géo).
- Essayer plusieurs méthodes d'imputation et comparer.

---

## Annexes (fichiers)
- `output_assets/result_cleaned.csv`
- `output_assets/anomalies.csv`
- `output_assets/report_assets/*`
- `output_assets/plots/*`
- `output_assets/model_joblib.pkl`

s
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from pathlib import Path

df = pd.read_csv('result.csv', sep=';', usecols=[
    # 'Id', # gain de performance en mémoire, passage de 1O.5kb à 9.5kb
    'gaming_interest_score',
    'insta_design_interest_score',
    'football_interest_score',
    'recommended_product',
    'campaign_success',
    'age',
    'canal_recommande'
])

# On enleve les lignes avec des valeurs nulles
# print(df.isnull().sum())
df = df.dropna()

# On affiche le nombre de doublons ( il y en aurait 4 si on enlèver l'id, mais faut les garder car c'est juste des gens qui ont le même type de profil)
# print(df.duplicated().sum())

#  On optimise les types de données pour réduire l'utilisation mémoire (int8 car age ou score forcémentre entre 0 et 100, donc -128 à 127
# df['Id'] = df['Id'].astype('int16') 
df['age'] = df['age'].astype('int16')
df['gaming_interest_score'] = df['gaming_interest_score'].astype('int16')
df['insta_design_interest_score'] = df['insta_design_interest_score'].astype('int16')
df['football_interest_score'] = df['football_interest_score'].astype('int16')
df['football_interest_score'] = df['football_interest_score'].astype('int16')

# Conversion de campaign_success en booléen
campaign_map = {
    'true': True, 
    'false': False,
}
df['campaign_success'] = (
    df['campaign_success']
    .astype(str)
    .str.lower()
    .map(campaign_map)
)

df['canal_recommande'] = df['canal_recommande'].str.strip().str.lower()
df['recommended_product'] = df['recommended_product'].str.strip().str.lower()
# Corriger les fautes de fornite
df['recommended_product'] = df['recommended_product'].replace({'fornite': 'fortnite'})

# Filtrer les valeurs non souhaitées avant toute statistique
# Garder UNIQUEMENT facebook, mail, insta pour canal_recommande
# Garder UNIQUEMENT fifa, fortnite, instagram pack pour recommended_product
allowed_supports = ['facebook', 'mail', 'insta']
allowed_products = ['fifa', 'fortnite', 'instagram pack']
df = df[df['canal_recommande'].isin(allowed_supports) & df['recommended_product'].isin(allowed_products)]

# Convertir en category après filtrage
df['recommended_product'] = df['recommended_product'].astype('category')
df['canal_recommande'] = df['canal_recommande'].astype('category')

print(f"\nAprès filtrage des valeurs invalides:")
print(f"  Lignes conservées: {len(df)}")
print(f"  Valeurs canal_recommande: {df['canal_recommande'].unique().tolist()}")
print(f"  Valeurs recommended_product: {df['recommended_product'].unique().tolist()}")

# ===================================================================
# DÉTECTION DES ANOMALIES
# ===================================================================
# Objectif : Identifier et écarter les points atypiques susceptibles de
#            fausser l'analyse (erreurs de mesure, comportements exceptionnels)
#
# Méthodes statistiques simples (basées sur le cours) :
#
# 1) ÉCART-TYPE (3σ) :
#    Une valeur est considérée comme anormale si elle est trop loin de la moyenne.
#    Seuil : |valeur - moyenne| > 3 × écart-type
#
# 2) INTERQUARTILE RANGE (IQR) :
#    Basée sur le 1er et 3e quartile (Q1, Q3).
#    IQR = Q3 - Q1
#    Anomalie si : valeur < Q1 - 1.5·IQR  OU  valeur > Q3 + 1.5·IQR
#
# Justification :
# - Méthodes simples, robustes et complémentaires
# - L'écart-type détecte les valeurs extrêmes par rapport à la moyenne globale (pas pertinent dans notre )
# - IQR est plus robuste aux valeurs extrêmes (utilise les quartiles)
# ===================================================================

# Liste des colonnes de scores à analyser
scores_cols = ['gaming_interest_score', 'insta_design_interest_score', 'football_interest_score']

# Dictionnaire pour stocker les statistiques de chaque score
stats = {}

print("\n" + "="*70)
print("DÉTECTION DES ANOMALIES - TOUS LES SCORES D'INTÉRÊT")
print("="*70)

# Initialiser les colonnes d'anomalies globales
df['anomalie_std_global'] = False
df['anomalie_iqr_global'] = False

for col in scores_cols:
    print(f"\n{'='*70}")
    print(f"ANALYSE : {col.upper()}")
    print(f"{'='*70}")
    
    # Méthode 1 : Écart-type (3σ)
    moyenne = df[col].mean()
    ecart_type = df[col].std()
    
    anomalies_std = df[(df[col] > moyenne + 3*ecart_type) |
                       (df[col] < moyenne - 3*ecart_type)]
    
    print(f"\nMéthode 1 - Écart-Type (3σ):")
    print(f"  Moyenne: {moyenne:.2f}")
    print(f"  Écart-type: {ecart_type:.2f}")
    print(f"  Borne inférieure: {moyenne - 3*ecart_type:.2f}")
    print(f"  Borne supérieure: {moyenne + 3*ecart_type:.2f}")
    print(f"  → Anomalies détectées: {len(anomalies_std)} / {len(df)} ({100*len(anomalies_std)/len(df):.1f}%)")
    
    # Méthode 2 : IQR (Interquartile Range)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    anomalies_iqr = df[(df[col] < Q1 - 1.5*IQR) | 
                       (df[col] > Q3 + 1.5*IQR)]
    
    print(f"\nMéthode 2 - IQR (Interquartile Range):")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Borne inférieure: {Q1 - 1.5*IQR:.2f}")
    print(f"  Borne supérieure: {Q3 + 1.5*IQR:.2f}")
    print(f"  → Anomalies détectées: {len(anomalies_iqr)} / {len(df)} ({100*len(anomalies_iqr)/len(df):.1f}%)")
    
    # Marquer les anomalies dans le DataFrame
    df[f'anomalie_std_{col}'] = (df[col] > moyenne + 3*ecart_type) | \
                                 (df[col] < moyenne - 3*ecart_type)
                         
    df[f'anomalie_iqr_{col}'] = (df[col] < Q1 - 1.5*IQR) | \
                                 (df[col] > Q3 + 1.5*IQR)
    
    # Anomalie combinée (détectée par au moins une méthode)
    df[f'anomalie_{col}'] = df[f'anomalie_std_{col}'] | df[f'anomalie_iqr_{col}']
    
    # Mettre à jour les anomalies globales
    df['anomalie_std_global'] = df['anomalie_std_global'] | df[f'anomalie_std_{col}']
    df['anomalie_iqr_global'] = df['anomalie_iqr_global'] | df[f'anomalie_iqr_{col}']
    
    print(f"\nRésumé combiné ({col}):")
    print(f"  Total anomalies: {df[f'anomalie_{col}'].sum()} ({100*df[f'anomalie_{col}'].sum()/len(df):.1f}%)")
    print(f"  Points normaux: {(~df[f'anomalie_{col}']).sum()} ({100*(~df[f'anomalie_{col}']).sum()/len(df):.1f}%)")
    
    # Stocker les statistiques pour les visualisations
    stats[col] = {
        'moyenne': moyenne,
        'ecart_type': ecart_type,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }

# Anomalie globale (détectée dans au moins un score par méthodes statistiques)
df['anomalie_globale'] = df['anomalie_std_global'] | df['anomalie_iqr_global']

print(f"\n{'='*70}")
print("RÉSUMÉ GLOBAL - TOUS LES SCORES (MÉTHODES STATISTIQUES)")
print(f"{'='*70}")
print(f"  Lignes avec au moins 1 anomalie: {df['anomalie_globale'].sum()} ({100*df['anomalie_globale'].sum()/len(df):.1f}%)")
print(f"  Lignes sans anomalie: {(~df['anomalie_globale']).sum()} ({100*(~df['anomalie_globale']).sum()/len(df):.1f}%)")
print("="*70 + "\n")

# ===================================================================
# MÉTHODE 3 : DÉTECTION PAR PLAGES VALIDES
# ===================================================================
# Détection simple basée sur des plages de valeurs acceptables :
# - Age : entre 20 et 100 inclus
# - Scores : entre 0 et 100 inclus
# ===================================================================

print("\n" + "="*70)
print("MÉTHODE 3 - DÉTECTION PAR PLAGES VALIDES")
print("="*70)

# Détection pour l'âge
df['anomalie_age_plage'] = (df['age'] < 20) | (df['age'] > 100)
anomalies_age = df[df['anomalie_age_plage']]

print(f"\nAge (plage valide: 20-100 inclus):")
print(f"  → Anomalies détectées: {len(anomalies_age)} / {len(df)} ({100*len(anomalies_age)/len(df):.1f}%)")
if len(anomalies_age) > 0:
    print(f"  Valeurs anormales: {sorted(anomalies_age['age'].unique())}")

# Détection pour les scores (plage 0-100)
df['anomalie_gaming_plage'] = (df['gaming_interest_score'] < 0) | (df['gaming_interest_score'] > 100)
df['anomalie_insta_plage'] = (df['insta_design_interest_score'] < 0) | (df['insta_design_interest_score'] > 100)
df['anomalie_football_plage'] = (df['football_interest_score'] < 0) | (df['football_interest_score'] > 100)

print(f"\nGaming Interest Score (plage valide: 0-100 inclus):")
print(f"  → Anomalies détectées: {df['anomalie_gaming_plage'].sum()} / {len(df)} ({100*df['anomalie_gaming_plage'].sum()/len(df):.1f}%)")

print(f"\nInsta Design Interest Score (plage valide: 0-100 inclus):")
print(f"  → Anomalies détectées: {df['anomalie_insta_plage'].sum()} / {len(df)} ({100*df['anomalie_insta_plage'].sum()/len(df):.1f}%)")

print(f"\nFootball Interest Score (plage valide: 0-100 inclus):")
print(f"  → Anomalies détectées: {df['anomalie_football_plage'].sum()} / {len(df)} ({100*df['anomalie_football_plage'].sum()/len(df):.1f}%)")

# Anomalie de plage combinée
df['anomalie_plage_global'] = (df['anomalie_age_plage'] | 
                                df['anomalie_gaming_plage'] | 
                                df['anomalie_insta_plage'] | 
                                df['anomalie_football_plage'])

print(f"\nRésumé - Méthode Plages:")
print(f"  Total lignes avec anomalie de plage: {df['anomalie_plage_global'].sum()} ({100*df['anomalie_plage_global'].sum()/len(df):.1f}%)")
print("="*70 + "\n")

# ===================================================================
# ANOMALIE FINALE (TOUTES MÉTHODES COMBINÉES)
# ===================================================================
df['anomalie_finale'] = df['anomalie_globale'] | df['anomalie_plage_global']

print("\n" + "="*70)
print("RÉSUMÉ FINAL - TOUTES MÉTHODES COMBINÉES")
print("="*70)
print(f"  Total lignes avec AU MOINS 1 anomalie: {df['anomalie_finale'].sum()} ({100*df['anomalie_finale'].sum()/len(df):.1f}%)")
print(f"  Lignes sans aucune anomalie: {(~df['anomalie_finale']).sum()} ({100*(~df['anomalie_finale']).sum()/len(df):.1f}%)")
print("="*70 + "\n")

# ===================================================================
# PHASE 3 : ANALYSE STATISTIQUE & KPI
# ===================================================================
# Objectif : comprendre les profils avant toute prédiction.
# - Calcul des KPI clefs (global, produit, segment d'intérêt, support, âge)
# - Visualisations simples pour le rapport
# - Matrice de corrélation pour identifier les relations entre variables
# ===================================================================

plots_dir = Path("output_assets/plots")
report_dir = Path("output_assets/report_assets")
plots_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

# On exclut les anomalies pour l'analyse descriptive afin de ne pas biaiser les KPI.
df_kpi = df[~df['anomalie_finale']].copy()

def success_rate(series: pd.Series) -> float:
    return 100 * series.mean() if len(series) else float("nan")

print("\n" + "="*70)
print("PHASE 3 - ANALYSE STATISTIQUE ET KPI (données sans anomalies)")
print("="*70)
print(f"Lignes retenues pour l'analyse: {len(df_kpi)} / {len(df)}")

# KPI globaux
global_success = success_rate(df_kpi['campaign_success'])
kpi_summary = pd.DataFrame([
    {"kpi": "global_success_rate_pct", "value": round(global_success, 2)},
    {"kpi": "rows_used", "value": len(df_kpi)}
])
kpi_summary.to_csv("kpi_summary.csv", index=False)

print(f"Taux de réussite global: {global_success:.2f}%")

# KPI par produit recommandé
kpi_product = df_kpi.groupby('recommended_product', observed=False)['campaign_success'].agg([
    ('success_rate_pct', lambda x: success_rate(x)),
    ('n', 'size')
]).sort_values('success_rate_pct', ascending=False)
kpi_product.to_csv(report_dir / 'kpi_success_by_product.csv')

# KPI par segment d'intérêt (dominant sur les 3 scores)
interest_cols = ['gaming_interest_score', 'insta_design_interest_score', 'football_interest_score']
df_kpi['segment_interet'] = df_kpi[interest_cols].idxmax(axis=1).str.replace('_interest_score', '', regex=False)
kpi_segment = df_kpi.groupby('segment_interet', observed=False)['campaign_success'].agg([
    ('success_rate_pct', lambda x: success_rate(x)),
    ('n', 'size')
]).sort_values('success_rate_pct', ascending=False)

# KPI par support / canal
kpi_support = df_kpi.groupby('canal_recommande', observed=False)['campaign_success'].agg([
    ('success_rate_pct', lambda x: success_rate(x)),
    ('n', 'size')
]).sort_values('success_rate_pct', ascending=False)
kpi_support.to_csv(report_dir / 'kpi_success_by_support.csv')

# KPI par tranche d'âge
age_bins = [0, 24, 34, 44, 54, 64, 120]
age_labels = ['20-24', '25-34', '35-44', '45-54', '55-64', '65+']
df_kpi['age_group'] = pd.cut(df_kpi['age'], bins=age_bins, labels=age_labels, right=True, include_lowest=True)
kpi_age = df_kpi.groupby('age_group', observed=False)['campaign_success'].agg([
    ('success_rate_pct', lambda x: success_rate(x)),
    ('n', 'size')
]).sort_values('success_rate_pct', ascending=False)
kpi_age.to_csv(report_dir / 'kpi_success_by_agegroup.csv')

# Export par segment d'intérêt pour consultation ultérieure
kpi_segment.to_csv(report_dir / 'kpi_success_by_interest.csv')

print("\nTop produits (taux de réussite) :")
print(kpi_product.head(5).to_string())
print("\nSupports les plus performants :")
print(kpi_support.head(5).to_string())
print("\nTranches d'âge les plus réceptives :")
print(kpi_age.head(5).to_string())
print("\nSegments d'intérêt :")
print(kpi_segment.head(5).to_string())

# Fonctions utilitaires pour les barplots
def plot_bar(dataframe, title, xlabel, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    dataframe['success_rate_pct'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_ylabel('Taux de réussite (%)')
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=120)
    plt.show()

# Graphiques KPI
plot_bar(kpi_product, "Taux de réussite par produit", "Produit", "kpi_success_by_product.png")
plot_bar(kpi_support, "Taux de réussite par support", "Support", "kpi_success_by_support.png")
plot_bar(kpi_age, "Taux de réussite par tranche d'âge", "Tranche d'âge", "kpi_success_by_agegroup.png")
plot_bar(kpi_segment, "Taux de réussite par segment d'intérêt", "Segment", "kpi_success_by_interest.png")

# Corrélations entre KPI de base (scores, âge, succès)
print("\n" + "="*70)
print("Matrice de corrélation (scores & succès)")
print("="*70)
df_corr = df_kpi[interest_cols + ['age', 'campaign_success']].copy()
df_corr['campaign_success'] = df_corr['campaign_success'].astype(int)
corr_matrix = df_corr.corr()
print(corr_matrix.round(3))

# Visualisation de la matrice de corrélation
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr_matrix.index)))
ax.set_yticklabels(corr_matrix.index)
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_title('Corrélation entre scores, âge et succès', fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'correlation_matrix.png', dpi=120)
plt.show()

# Synthèse rapide à inclure dans le rapport
def describe_best(kpi_df, label):
    if len(kpi_df) == 0:
        return f"Aucun {label} disponible."
    best = kpi_df.iloc[0]
    return f"Meilleur {label}: {kpi_df.index[0]} avec {best['success_rate_pct']:.1f}% (n={int(best['n'])})."

print("\n" + "-"*70)
print("SYNTHÈSE RAPIDE POUR LE RAPPORT")
print("-"*70)
print(describe_best(kpi_product, "produit"))
print(describe_best(kpi_support, "support"))
print(describe_best(kpi_age, "tranche d'âge"))
print(describe_best(kpi_segment, "segment d'intérêt"))
print("Hypothèses à creuser : tester des messages ciblés sur les segments et supports les plus performants ; vérifier la robustesse de ces tendances sur un échantillon plus large.")

# ===================================================================
# ÉNUMÉRATION DES LIGNES AVEC ANOMALIES
# ===================================================================
print("\n" + "="*70)
print("LISTE DES LIGNES CONTENANT AU MOINS UNE ANOMALIE")
print("="*70)

lignes_anomalies = df[df['anomalie_finale']].copy()

if len(lignes_anomalies) > 0:
    print(f"\nNombre total de lignes avec anomalies: {len(lignes_anomalies)}\n")
    
    # Afficher les informations clés pour chaque ligne avec anomalie
    print("Index | Age | Gaming | Insta | Football | Type(s) d'anomalie(s)")
    print("-" * 70)
    
    for idx in lignes_anomalies.index:
        age = df.loc[idx, 'age']
        gaming = df.loc[idx, 'gaming_interest_score']
        insta = df.loc[idx, 'insta_design_interest_score']
        football = df.loc[idx, 'football_interest_score']
        
        # Identifier les types d'anomalies pour cette ligne
        types_anomalies = []
        
        if df.loc[idx, 'anomalie_age_plage']:
            types_anomalies.append(f"Age hors plage")
        
        if df.loc[idx, 'anomalie_gaming_plage']:
            types_anomalies.append(f"Gaming hors plage")
        elif df.loc[idx, 'anomalie_gaming_interest_score']:
            types_anomalies.append(f"Gaming (stat)")
        
        if df.loc[idx, 'anomalie_insta_plage']:
            types_anomalies.append(f"Insta hors plage")
        elif df.loc[idx, 'anomalie_insta_design_interest_score']:
            types_anomalies.append(f"Insta (stat)")
        
        if df.loc[idx, 'anomalie_football_plage']:
            types_anomalies.append(f"Football hors plage")
        elif df.loc[idx, 'anomalie_football_interest_score']:
            types_anomalies.append(f"Football (stat)")
        
        print(f"{idx:5d} | {age:3d} | {gaming:6d} | {insta:5d} | {football:8d} | {', '.join(types_anomalies)}")
    
    print("\n" + "="*70)
    print("Détail des premières lignes avec anomalies:")
    print("="*70)
    # Afficher un échantillon plus détaillé (les 10 premières)
    cols_to_show = ['age', 'gaming_interest_score', 'insta_design_interest_score', 
                    'football_interest_score', 'recommended_product', 'campaign_success']
    print(lignes_anomalies[cols_to_show].head(10).to_string())
    print(f"\n... ({len(lignes_anomalies)} lignes au total avec anomalies)\n")
else:
    print("\nAucune anomalie détectée dans les données.\n")

print("="*70 + "\n")

# ===================================================================
# VISUALISATIONS (MÉTHODES STATISTIQUES)
# ===================================================================
# Pour chaque type de données, on trace :
# - Valeurs normales en BLEU
# - Anomalies en ROUGE (détectées par méthodes statistiques)
# - Moyenne et bornes en POINTILLÉS
# ===================================================================

print("Génération des graphiques de visualisation...\n")

# Ajouter un léger jitter pour mieux voir les points superposés
np.random.seed(42)
jitter = np.random.normal(0, 0.1, size=len(df))
ages_jitter = df['age'].values + jitter

# Créer les visualisations pour chaque score
for col in scores_cols:
    # Récupérer les statistiques
    moyenne = stats[col]['moyenne']
    ecart_type = stats[col]['ecart_type']
    Q1 = stats[col]['Q1']
    Q3 = stats[col]['Q3']
    IQR = stats[col]['IQR']
    
    # Nom lisible pour les titres
    col_name = col.replace('_', ' ').title()
    
    # ---------------------------------------------------------------
    # Figure 1 : Vue d'ensemble avec méthode Écart-Type (3σ)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tracer les points normaux en bleu
    normaux = df[~df[f'anomalie_std_{col}']]
    ax.scatter(ages_jitter[~df[f'anomalie_std_{col}']], normaux[col], 
               c='blue', alpha=0.5, s=30, label='Valeurs normales', edgecolors='none')
    
    # Tracer les anomalies en rouge
    anomalies = df[df[f'anomalie_std_{col}']]
    ax.scatter(ages_jitter[df[f'anomalie_std_{col}']], anomalies[col], 
               c='red', alpha=0.8, s=50, label='Anomalies', edgecolors='black', linewidths=1)
    
    # Lignes de référence (moyenne et bornes)
    ax.axhline(moyenne, color='green', linestyle='--', linewidth=2, label=f'Moyenne ({moyenne:.2f})')
    ax.axhline(moyenne + 3*ecart_type, color='orange', linestyle=':', linewidth=2, 
               label=f'Bornes ±3σ ({moyenne - 3*ecart_type:.2f} / {moyenne + 3*ecart_type:.2f})')
    ax.axhline(moyenne - 3*ecart_type, color='orange', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel(col_name, fontsize=12)
    ax.set_title(f'Détection d\'anomalies - {col_name} - Méthode Écart-Type (3σ)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------------
    # Figure 2 : Vue d'ensemble avec méthode IQR
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Tracer les points normaux en bleu
    normaux_iqr = df[~df[f'anomalie_iqr_{col}']]
    ax.scatter(ages_jitter[~df[f'anomalie_iqr_{col}']], normaux_iqr[col], 
               c='blue', alpha=0.5, s=30, label='Valeurs normales', edgecolors='none')
    
    # Tracer les anomalies en rouge
    anomalies_iqr_pts = df[df[f'anomalie_iqr_{col}']]
    ax.scatter(ages_jitter[df[f'anomalie_iqr_{col}']], anomalies_iqr_pts[col], 
               c='red', alpha=0.8, s=50, label='Anomalies', edgecolors='black', linewidths=1)
    
    # Lignes de référence (quartiles et bornes IQR)
    ax.axhline(Q1, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Q1 ({Q1:.2f})')
    ax.axhline(Q3, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Q3 ({Q3:.2f})')
    ax.axhline(Q1 - 1.5*IQR, color='orange', linestyle=':', linewidth=2, 
               label=f'Bornes IQR ({Q1 - 1.5*IQR:.2f} / {Q3 + 1.5*IQR:.2f})')
    ax.axhline(Q3 + 1.5*IQR, color='orange', linestyle=':', linewidth=2)
    ax.axhline(moyenne, color='green', linestyle='--', linewidth=2, alpha=0.5, label=f'Moyenne ({moyenne:.2f})')
    
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel(col_name, fontsize=12)
    ax.set_title(f'Détection d\'anomalies - {col_name} - Méthode IQR', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------------
    # Figure 3 : Zoom sur une zone spécifique pour examiner finement les anomalies
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    
    zoom_min = max(0, moyenne - 2.5*ecart_type)
    zoom_max = moyenne + 2.5*ecart_type
    
    # Filtrer les données dans la zone de zoom
    mask_zoom = (df[col] >= zoom_min) & (df[col] <= zoom_max)
    df_zoom = df[mask_zoom]
    ages_jitter_zoom = ages_jitter[mask_zoom]
    
    # Tracer normaux et anomalies dans cette zone
    normaux_zoom = df_zoom[~df_zoom[f'anomalie_{col}']]
    anomalies_zoom = df_zoom[df_zoom[f'anomalie_{col}']]
    
    ax.scatter(ages_jitter_zoom[~df_zoom[f'anomalie_{col}']], normaux_zoom[col], 
               c='blue', alpha=0.5, s=40, label='Valeurs normales', edgecolors='none')
    ax.scatter(ages_jitter_zoom[df_zoom[f'anomalie_{col}']], anomalies_zoom[col], 
               c='red', alpha=0.9, s=60, label='Anomalies', edgecolors='black', linewidths=1.5)
    
    ax.axhline(moyenne, color='green', linestyle='--', linewidth=2, label=f'Moyenne ({moyenne:.2f})')
    ax.axhline(moyenne + 3*ecart_type, color='orange', linestyle=':', linewidth=2, label='Bornes ±3σ')
    ax.axhline(moyenne - 3*ecart_type, color='orange', linestyle=':', linewidth=2)
    ax.axhline(Q1 - 1.5*IQR, color='purple', linestyle=':', linewidth=1.5, alpha=0.6, label='Bornes IQR')
    ax.axhline(Q3 + 1.5*IQR, color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel(col_name, fontsize=12)
    ax.set_title(f'Zoom - {col_name} : Examen fin des anomalies (zone {zoom_min:.1f} - {zoom_max:.1f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(zoom_min - 0.5, zoom_max + 0.5)
    plt.tight_layout()
    plt.show()


# # get types of columns
print(df.info())
print(df.head())
del df
gc.collect()
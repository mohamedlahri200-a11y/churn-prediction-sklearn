# Prédiction du churn client (Telco) avec scikit-learn

Projet d'autoformation en machine learning : prédire quels clients d'un opérateur télécom risquent de résilier leur abonnement, afin de permettre des actions de rétention ciblées.

## Problématique
Un client perdu coûte plus cher qu'un client conservé. L'objectif est d'identifier à l'avance les clients à risque (classification binaire : *Reste* / *Parti*).

## Données
- Jeu de données : IBM Telco Customer Churn (7043 clients, 33 colonnes à l'origine).
- Cible : `Churn Value` (1 = parti), environ 26,5 % de départs (classes déséquilibrées).
- Le fichier CSV n'est pas versionné : place-le dans `data/Telco_customer_churn.csv`
  (séparateur `;`, décimale `,`).

## Méthodologie
1. **Nettoyage** : conversion de `Total Charges` en numérique (11 valeurs vides, clients avec ancienneté 0).
2. **EDA** : analyse de la cible, ancienneté, type de contrat, mode de paiement, service internet.
3. **Prévention de la fuite de données** : suppression de `Churn Label`, `Churn Score`, `Churn Reason`, `CLTV`, ainsi que des colonnes d'identifiant, constantes et géographiques.
4. **Préparation** : séparation train/test stratifiée (80/20), `ColumnTransformer` (StandardScaler + OneHotEncoder) dans un `Pipeline`.
5. **Modèles** : baseline (Dummy), régression logistique, version `class_weight='balanced'`, comparaison par validation croisée stratifiée (5 plis) avec Random Forest et Gradient Boosting.
6. **Choix du seuil** de décision (0.55) par validation croisée sur le train, puis évaluation finale unique sur le test.

## Résultats (jeu de test, 1409 clients)

| Modèle | Accuracy | Precision (Parti) | Recall (Parti) | F1 (Parti) |
|---|---|---|---|---|
| Baseline « toujours reste » | 0.735 | - | 0.00 | - |
| Régression logistique (seuil 0.5) | 0.80 | 0.64 | 0.57 | 0.61 |
| Régression logistique équilibrée, seuil 0.55 (retenu) | 0.76 | 0.53 | 0.74 | 0.62 |

**Pourquoi ce modèle ?** Dans un contexte de rétention, rater un client qui part (faux négatif) coûte plus cher que contacter un client à tort (faux positif). On privilégie donc le recall, au prix d'une accuracy plus faible.

## Facteurs influents (coefficients de la régression logistique)
- **Augmentent le risque de départ** : service Fibre optique, paiement par chèque électronique, services de streaming.
- **Réduisent le risque** : contrat de 1 ou 2 ans, ancienneté élevée, présence de personnes à charge.
- Les variables `Tenure`, `Monthly Charges` et `Total Charges` sont corrélées entre elles (multicollinéarité) : leurs coefficients individuels ne sont pas interprétés.

## Limites
- Les coefficients montrent des **associations**, pas des causes.
- Le seuil de 0.55 est un compromis ; il doit être ajusté selon la capacité d'action du service client.
- Pas de réglage poussé des hyperparamètres ni de modèles avancés (pistes d'amélioration).

## Lancer le projet
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Puis ouvrir `01_exploration.ipynb` et exécuter toutes les cellules.

## Stack
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, joblib.
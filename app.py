import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Prédiction du churn", layout="wide")
st.title("Prédiction du churn client")

paquet = joblib.load('models/churn_logreg.joblib')
modele, seuil = paquet['modele'], paquet['seuil']
colonnes = list(modele.feature_names_in_)
ref = pd.read_csv('data/Telco_customer_churn.csv', sep=';', decimal=',')


def nettoyer(d):
    d = d.copy()
    d['Total Charges'] = pd.to_numeric(
        d['Total Charges'].astype(str).str.strip().str.replace(',', '.'),
        errors='coerce'
    ).fillna(0)
    return d


tab1, tab2 = st.tabs(["Un client", "Liste de clients (CSV)"])

# ---------- Mode 1 : un client ----------
with tab1:
    valeurs = {}
    cols = st.columns(3)
    i = 0
    for col in colonnes:
        if col == 'Total Charges':
            continue
        with cols[i % 3]:
            if col == 'Tenure Months':
                valeurs[col] = st.slider(col, 0, 72, 12)
            elif col == 'Monthly Charges':
                valeurs[col] = st.slider(col, 18.0, 120.0, 70.0)
            else:
                valeurs[col] = st.selectbox(col, sorted(ref[col].dropna().unique()))
        i += 1
    # Total Charges estimé : ancienneté x facture mensuelle
    valeurs['Total Charges'] = valeurs['Tenure Months'] * valeurs['Monthly Charges']
    client = pd.DataFrame([valeurs])[colonnes]

    if st.button("Prédire"):
        p = modele.predict_proba(client)[0, 1]
        st.metric("Probabilité de départ", f"{p:.0%}")
        if p >= seuil:
            st.error("Client à risque : action de rétention conseillée")
        else:
            st.success("Client plutôt fidèle")

# ---------- Mode 2 : liste de clients ----------
with tab2:
    st.write("Charge un CSV de clients (séparateur `;`, décimale `,`, mêmes colonnes que le dataset).")
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is not None:
        d = pd.read_csv(fichier, sep=';', decimal=',')
        manquantes = [c for c in colonnes if c not in d.columns]
        if manquantes:
            st.error(f"Colonnes manquantes : {manquantes}")
        else:
            d = nettoyer(d)
            d['Proba départ'] = modele.predict_proba(d[colonnes])[:, 1].round(3)
            d['À contacter'] = (d['Proba départ'] >= seuil).map({True: 'Oui', False: 'Non'})
            d = d.sort_values('Proba départ', ascending=False)

            c1, c2 = st.columns(2)
            c1.metric("Clients analysés", len(d))
            c2.metric("Clients à contacter", int((d['À contacter'] == 'Oui').sum()))

            affichage = [c for c in ['CustomerID', 'Contract', 'Tenure Months',
                                     'Monthly Charges', 'Proba départ', 'À contacter']
                         if c in d.columns]
            st.dataframe(d[affichage].head(50), use_container_width=True)

            st.download_button(
                "Télécharger la liste complète (CSV)",
                d.to_csv(index=False, sep=';').encode('utf-8'),
                file_name="clients_scores.csv",
                mime="text/csv",
            )
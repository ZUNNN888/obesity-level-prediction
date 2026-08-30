from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Obesity Level Prediction",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.7rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(14,116,144,.16), rgba(15,118,110,.10));
        border: 1px solid rgba(14,116,144,.18);
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.35rem;}
    .hero p {margin: 0; font-size: 1.02rem; opacity: .85;}
    .info-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        height: 100%;
    }
    .result-box {
        padding: 1.25rem 1.3rem;
        border-radius: 15px;
        border: 1px solid rgba(14,116,144,.25);
        background: rgba(14,116,144,.07);
    }
    .small-note {font-size: .9rem; opacity: .76;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.18); padding: .75rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_obesity_data():
    """Load the cleaned UCI Obesity Levels dataset bundled with this app."""
    data = pd.read_csv(BASE_DIR / "ObesityDataSet_cleaned.csv")
    return data.reset_index(drop=True)


@st.cache_resource
def load_model():
    """Rebuild the deployed champion model from the cleaned coursework dataset.

    Training at startup avoids binary model compatibility issues on Community Cloud.
    The bundled cleaned CSV, split, and hyperparameters match the evaluation setup
    documented in the report.
    """
    data = load_obesity_data()
    X = data.drop(columns=["NObeyesdad"])
    y = data["NObeyesdad"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [
        "Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"
    ]
    categorical_features = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC",
        "SMOKE", "SCC", "CALC", "MTRANS"
    ]

    def make_pipeline(estimator):
        prep = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )
        return Pipeline([("prep", prep), ("model", estimator)])

    lr = make_pipeline(
        LogisticRegression(C=20, class_weight="balanced", max_iter=3000)
    )
    knn = make_pipeline(
        KNeighborsClassifier(n_neighbors=5, leaf_size=50, p=1, weights="distance")
    )
    rf = make_pipeline(
        RandomForestClassifier(
            n_estimators=600,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=0.5,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    )

    model = VotingClassifier(
        estimators=[("lr", lr), ("knn", knn), ("rf", rf)],
        voting="soft",
        weights=[2, 1, 3],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


@st.cache_data
def load_assets():
    data = load_obesity_data()
    metrics = pd.read_csv(BASE_DIR / "model_comparison.csv")
    importance = pd.read_csv(BASE_DIR / "feature_importance.csv")
    with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return data, metrics, importance, metadata


model = load_model()
df, metrics_df, feature_importance_df, metadata = load_assets()

CLASS_LABELS = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

FRIENDLY_CLASS = {
    "Insufficient_Weight": "Insufficient Weight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight Level I",
    "Overweight_Level_II": "Overweight Level II",
    "Obesity_Type_I": "Obesity Type I",
    "Obesity_Type_II": "Obesity Type II",
    "Obesity_Type_III": "Obesity Type III",
}

MODEL_TRADEOFFS = pd.DataFrame(
    [
        ["Logistic Regression", "High", "Fast", "Linear decision boundaries", "Strong baseline and probability estimates"],
        ["KNN", "Medium", "Slower at prediction", "Sensitive to scaling and local density", "Useful similarity-based comparison"],
        ["Decision Tree", "Very high", "Very fast", "Can overfit and be split-sensitive", "Clear rules; strongest single test accuracy"],
        ["Random Forest", "Medium", "Moderate", "Larger model and less transparent", "Stable non-linear ensemble"],
        ["Soft Voting", "Medium", "Moderate", "More computation and complexity", "Best cross-validation stability and probability quality"],
    ],
    columns=["Model", "Interpretability", "Speed", "Main Trade-off", "Why Included"],
)


def hero(title: str, subtitle: str):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.title("⚖️ Obesity Analytics")
    page = st.radio(
        "Navigation",
        ["Overview", "Predict Obesity Level", "Model Comparison", "Data Explorer", "Methodology"],
    )
    st.divider()
    st.caption("BMDS2003 Data Science · CRISP-DM deployment prototype")
    st.info("Academic prototype only. It does not provide a medical diagnosis.")


if page == "Overview":
    hero(
        "Obesity Level Prediction",
        "A machine-learning prototype that estimates seven obesity-level categories from demographic, physical, eating-habit and lifestyle information.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique records", f"{metadata['unique_records']:,}")
    c2.metric("Input features", "16")
    c3.metric("Target classes", "7")
    champion_row = metrics_df.loc[metrics_df["Model"] == metadata["champion_model"]].iloc[0]
    c4.metric("Champion CV Macro F1", f"{champion_row['CV F1 Mean']:.3f}")

    st.subheader("Project objective")
    st.write(
        "The project compares multiple classification algorithms and deploys the selected model as an interactive Streamlit tool. "
        "The target is `NObeyesdad`, which contains seven classes from Insufficient Weight to Obesity Type III."
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="info-card"><b>1 · Understand</b><br><span class="small-note">Audit the UCI dataset, class distribution, features and data quality.</span></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="info-card"><b>2 · Model & Compare</b><br><span class="small-note">Tune five classifiers and compare accuracy, macro F1, ROC-AUC, log loss and CV stability.</span></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="info-card"><b>3 · Deploy</b><br><span class="small-note">Use the Soft Voting model for interactive prediction and probability output.</span></div>', unsafe_allow_html=True)

    st.subheader("Dataset provenance")
    st.write(
        "The UCI dataset contains observations from Colombia, Peru and Mexico. The source publication reports that 23% of the records were collected from users and 77% were synthetically generated during class balancing."
    )
    st.warning(
        "Important limitation: the original obesity labels were created using BMI-based classification rules. Therefore, height and weight are naturally very strong predictors and model performance should not be interpreted as independent clinical diagnosis."
    )

elif page == "Predict Obesity Level":
    hero(
        "Predict Obesity Level",
        "Enter one profile below. The same preprocessing pipeline used during model training is applied automatically.",
    )

    with st.form("prediction_form"):
        st.markdown("### A. Personal and physical information")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        age = c2.number_input("Age (years)", min_value=14.0, max_value=80.0, value=24.0, step=1.0)
        height = c3.number_input("Height (m)", min_value=1.30, max_value=2.20, value=1.70, step=0.01, format="%.2f")
        weight = c4.number_input("Weight (kg)", min_value=30.0, max_value=220.0, value=75.0, step=0.5)

        family_history = st.selectbox("Family history of overweight?", ["yes", "no"])

        st.markdown("### B. Eating habits")
        c1, c2, c3, c4 = st.columns(4)
        favc = c1.selectbox("Frequently eat high-calorie food?", ["yes", "no"])
        fcvc = c2.slider("Vegetable consumption (FCVC)", 1.0, 3.0, 2.0, 0.1, help="1 = low, 3 = high")
        ncp = c3.slider("Main meals per day (NCP)", 1.0, 4.0, 3.0, 0.1)
        caec = c4.selectbox("Food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])

        st.markdown("### C. Lifestyle and activity")
        c1, c2, c3, c4 = st.columns(4)
        smoke = c1.selectbox("Smoke?", ["no", "yes"])
        ch2o = c2.slider("Daily water consumption (CH2O)", 1.0, 3.0, 2.0, 0.1, help="Dataset scale: approximately 1 to 3")
        scc = c3.selectbox("Monitor calorie intake? (SCC)", ["no", "yes"])
        faf = c4.slider("Physical activity frequency (FAF)", 0.0, 3.0, 1.0, 0.1)

        c1, c2, c3 = st.columns(3)
        tue = c1.slider("Technology-use time (TUE)", 0.0, 2.0, 1.0, 0.1)
        calc = c2.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = c3.selectbox("Main transportation", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

        submitted = st.form_submit_button("Predict Obesity Level", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [{
                "Gender": gender,
                "Age": float(age),
                "Height": float(height),
                "Weight": float(weight),
                "family_history_with_overweight": family_history,
                "FAVC": favc,
                "FCVC": float(fcvc),
                "NCP": float(ncp),
                "CAEC": caec,
                "SMOKE": smoke,
                "CH2O": float(ch2o),
                "SCC": scc,
                "FAF": float(faf),
                "TUE": float(tue),
                "CALC": calc,
                "MTRANS": mtrans,
            }]
        )
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        prob_df = pd.DataFrame(
            {"Obesity Level": [FRIENDLY_CLASS[c] for c in model.classes_], "Probability": probabilities}
        ).sort_values("Probability", ascending=False)
        bmi = weight / (height ** 2)
        best_probability = float(prob_df.iloc[0]["Probability"])

        left, right = st.columns([1, 1.35])
        with left:
            st.markdown(
                f'<div class="result-box"><div class="small-note">Predicted category</div><h2>{FRIENDLY_CLASS[prediction]}</h2><b>Model probability: {best_probability:.1%}</b><br><br><span class="small-note">Reference BMI from entered height and weight: {bmi:.1f} kg/m²</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Probability is model confidence, not medical certainty.")
        with right:
            st.markdown("#### Prediction probabilities")
            chart_df = prob_df.set_index("Obesity Level")
            st.bar_chart(chart_df, horizontal=True)

        st.info(
            "Use this output for coursework demonstration and preliminary analytics only. A qualified healthcare professional should perform real health assessment and diagnosis."
        )

elif page == "Model Comparison":
    hero(
        "Model Comparison",
        "The report does not select a model from accuracy alone. It compares test performance, cross-validation stability, probability quality, interpretability and computational trade-offs.",
    )

    display = metrics_df.copy()
    score_cols = ["Accuracy", "Balanced Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "F1 (Weighted)", "ROC-AUC (Macro OVR)", "CV F1 Mean", "CV F1 Std"]
    for col in score_cols:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    display["Log Loss"] = display["Log Loss"].map(lambda x: f"{x:.3f}")
    display["Fit Time (s)"] = display["Fit Time (s)"].map(lambda x: f"{x:.3f}")
    display["Predict Time (s)"] = display["Predict Time (s)"].map(lambda x: f"{x:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Cross-validation comparison")
    cv_chart = metrics_df.set_index("Model")[["CV F1 Mean"]].sort_values("CV F1 Mean", ascending=False)
    st.bar_chart(cv_chart)

    st.markdown("### Algorithm trade-offs")
    st.dataframe(MODEL_TRADEOFFS, use_container_width=True, hide_index=True)

    tree = metrics_df.loc[metrics_df["Model"] == "Decision Tree"].iloc[0]
    soft = metrics_df.loc[metrics_df["Model"] == "Soft Voting"].iloc[0]
    st.success(
        f"Champion: Soft Voting. Decision Tree achieved the highest single test accuracy ({tree['Accuracy']:.3f}), "
        f"but Soft Voting achieved much stronger 5-fold CV Macro F1 ({soft['CV F1 Mean']:.3f} vs {tree['CV F1 Mean']:.3f}), "
        f"higher macro ROC-AUC ({soft['ROC-AUC (Macro OVR)']:.3f}) and lower log loss ({soft['Log Loss']:.3f}). "
        "This makes the ensemble a more defensible deployment choice."
    )

elif page == "Data Explorer":
    hero(
        "Data Explorer",
        "Explore the cleaned dataset used for modelling after removing 24 exact duplicate records.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Original rows", f"{metadata['original_records']:,}")
    c2.metric("Duplicates removed", f"{metadata['duplicates_removed']:,}")
    c3.metric("Rows used", f"{metadata['unique_records']:,}")

    st.markdown("### Target distribution")
    counts = df["NObeyesdad"].value_counts().rename_axis("Obesity Level").to_frame("Count")
    st.bar_chart(counts)

    st.markdown("### Numerical summary")
    numeric = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    st.dataframe(df[numeric].describe().T.round(3), use_container_width=True)

    st.markdown("### Feature importance from tuned Random Forest")
    imp = feature_importance_df.head(12).set_index("Feature")
    st.bar_chart(imp)
    st.caption("Feature importance describes this fitted model and does not establish causation.")

    with st.expander("View sample records"):
        st.dataframe(df.head(25), use_container_width=True)

elif page == "Methodology":
    hero(
        "Methodology & Deployment Notes",
        "The prototype follows the same data preparation and model-selection logic documented in the revised report.",
    )

    st.markdown("### CRISP-DM workflow")
    st.write(
        "**Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment.** "
        "The cleaned data is split 80/20 using stratification. Numeric features are standardised, categorical features are one-hot encoded, and all transformations are kept inside scikit-learn pipelines to reduce data leakage."
    )

    st.markdown("### Models evaluated")
    st.write("Multinomial Logistic Regression (baseline), KNN, Decision Tree, Random Forest and Soft Voting Ensemble.")

    st.markdown("### Hyperparameter tuning")
    st.write(
        "RandomizedSearchCV with 5-fold stratified cross-validation and macro F1 as the optimisation metric was used for the four individual models. The Soft Voting ensemble combines tuned Logistic Regression, KNN and Random Forest models."
    )

    st.markdown("### Why macro F1 matters")
    st.write(
        "The target has seven classes. Macro F1 gives each class equal weight, so a model cannot look strong merely by performing well on the most frequent category. Accuracy, weighted F1, macro ROC-AUC, log loss and confusion matrices are also considered."
    )

    st.markdown("### Limitations")
    st.write(
        "The dataset is not a clinical cohort: 77% of records were synthetically generated, the source population comes from Colombia, Peru and Mexico, and obesity labels were constructed using BMI-based rules. "
        "Consequently, the prototype demonstrates data-science methodology and should not be used as a stand-alone medical diagnostic tool."
    )

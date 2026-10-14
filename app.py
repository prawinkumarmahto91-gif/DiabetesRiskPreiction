import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------
# Diabetes Risk Prediction - Streamlit application
# Expected pipeline steps from diabetes.ipynb:
#   ("preprocessor", ColumnTransformer(...))
#   ("model", LogisticRegression(...))
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_logistic_regression_model.joblib"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.title("Diabetes Risk Prediction")
st.write(
    "Enter the feature values below to view the output of the saved "
    "machine-learning model."
)
st.warning(
    "Educational demonstration only — this model is not a medical "
    "diagnostic tool and must not be used to make healthcare decisions."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Place diabetes_logistic_regression_model.joblib in the same "
        "folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

if not hasattr(model, "named_steps"):
    st.error(
        "The saved model is not a scikit-learn Pipeline. "
        "Please save the fitted pipeline from your notebook."
    )
    st.stop()

step_names = list(model.named_steps.keys())
if "preprocessor" not in model.named_steps or "model" not in model.named_steps:
    st.error(
        "The model pipeline steps do not match your notebook. "
        f"Found steps: {step_names}. Expected 'preprocessor' and 'model'."
    )
    st.stop()

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

if not hasattr(model, "feature_names_in_"):
    st.error(
        "The saved model does not contain feature names. "
        "Retrain and save the complete pipeline using a pandas DataFrame."
    )
    st.stop()

feature_names = list(model.feature_names_in_)

# Read columns used by each fitted ColumnTransformer branch.
numeric_columns = []
categorical_columns = []
categorical_options = {}
numeric_default_values = {}

for transformer_name, transformer, columns in preprocessor.transformers_:
    if transformer_name == "remainder":
        continue

    if isinstance(columns, (list, tuple)):
        selected_columns = list(columns)
    else:
        # ColumnTransformer normally stores explicit column lists here
        try:
            selected_columns = list(columns)
        except TypeError:
            selected_columns = []

    normalized_name = transformer_name.lower()

    if "numeric" in normalized_name or "num" == normalized_name:
        numeric_columns.extend(selected_columns)

        # Start numeric fields at training-set means instead of all-zero values.
        # StandardScaler.mean_ is expressed in the original feature units.
        fitted_numeric_transformer = transformer
        if hasattr(fitted_numeric_transformer, "named_steps"):
            for substep in fitted_numeric_transformer.named_steps.values():
                if hasattr(substep, "mean_"):
                    fitted_numeric_transformer = substep
                    break

        if hasattr(fitted_numeric_transformer, "mean_"):
            for column, mean_value in zip(
                selected_columns, fitted_numeric_transformer.mean_
            ):
                numeric_default_values[column] = float(mean_value)

    if "categorical" in normalized_name or "category" in normalized_name or "cat" == normalized_name:
        categorical_columns.extend(selected_columns)

        fitted_transformer = transformer
        if hasattr(fitted_transformer, "named_steps"):
            # Also support a Pipeline that contains an encoder.
            for substep in fitted_transformer.named_steps.values():
                if hasattr(substep, "categories_"):
                    fitted_transformer = substep
                    break

        if hasattr(fitted_transformer, "categories_"):
            for column, options in zip(selected_columns, fitted_transformer.categories_):
                categorical_options[column] = [
                    value.item() if hasattr(value, "item") else value
                    for value in options
                ]

# Detect the preprocessing issue present in some versions of the notebook:
# every model input should be assigned to a preprocessing branch.
covered_columns = set(numeric_columns) | set(categorical_columns)
missing_from_preprocessing = [
    column for column in feature_names if column not in covered_columns
]

if missing_from_preprocessing:
    st.error(
        "The saved pipeline does not preprocess all input features. "
        "Your notebook's ColumnTransformer may have the categorical branch "
        "assigned to numeric_features instead of categorical_features. "
        "Fix that cell, rerun the preprocessing/model-training/save cells, "
        "and restart this app. Features not covered: "
        + ", ".join(missing_from_preprocessing)
    )
    st.stop()

# Guard against the duplicate numeric-column bug in the original notebook.
if len(numeric_columns) != len(set(numeric_columns)) or len(categorical_columns) != len(set(categorical_columns)):
    st.error(
        "Some features appear more than once in the preprocessing pipeline. "
        "In your notebook, check that the categorical transformer uses "
        "categorical_features, not numeric_features. Retrain and save the model."
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Input features")
    st.caption(
        "Use the same units and category labels as the dataset used to train the model."
    )

    input_values = {}
    numeric_set = set(numeric_columns)
    categorical_set = set(categorical_columns)

    # Two-column layout makes the 34 inputs easier to navigate.
    left_col, right_col = st.columns(2)

    for index, feature in enumerate(feature_names):
        target_col = left_col if index % 2 == 0 else right_col

        with target_col:
            label = feature.replace("_", " ")

            if feature in categorical_set:
                options = categorical_options.get(feature, [])
                if not options:
                    st.error(
                        f"No learned category options found for '{feature}'. "
                        "Check that the categorical branch uses OneHotEncoder "
                        "and retrain the model."
                    )
                    st.stop()

                input_values[feature] = st.selectbox(
                    label,
                    options=options,
                    key=f"input_{feature}"
                )
            elif feature in numeric_set:
                input_values[feature] = st.number_input(
                    label,
                    value=float(numeric_default_values.get(feature, 0.0)),
                    format="%.4f",
                    key=f"input_{feature}"
                )
            else:
                st.error(f"Feature '{feature}' is not assigned to a transformer.")
                st.stop()

    submitted = st.form_submit_button(
        "Generate model prediction",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    try:
        predicted_class = model.predict(input_df)[0]
        class_probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        st.subheader("Model output")

        positive_index = classes.index("Positive")
        positive_probability = float(class_probabilities[positive_index])

        if predicted_class == "Positive":
            st.warning("Model prediction: Positive")
        else:
            st.success("Model prediction: Negative")

        st.metric(
            label="Model-estimated Positive probability",
            value=f"{positive_probability:.2%}"
        )

        probability_table = pd.DataFrame({
            "Class": classes,
            "Model-estimated probability": class_probabilities
        })
        probability_table["Model-estimated probability"] = (
            probability_table["Model-estimated probability"].map(
                lambda value: f"{value:.2%}"
            )
        )
        st.dataframe(probability_table, hide_index=True, use_container_width=True)

        st.caption(
            "The probability is produced by the model and is not necessarily "
            "calibrated to real-world disease risk. This result is not a diagnosis."
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the model was retrained and saved "
            "after correcting the preprocessing columns. "
            f"Technical details: {exc}"
        )

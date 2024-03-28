import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from skactiveml.pool import UncertaintySampling, QueryByCommittee, RandomSampling, ProbabilisticAL, MonteCarloEER, CostEmbeddingAL
from xgboost import XGBClassifier
from skactiveml.classifier import SklearnClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from skactiveml.utils import MISSING_LABEL

import warnings
warnings.filterwarnings("ignore")

def run_active_learning(X, y, X_test, y_test, clf, strategy, rounds, R=False, metric=accuracy_score):

    model = SklearnClassifier(
        GaussianProcessClassifier(random_state=0),
        classes=np.unique(y),
        random_state=0
    )
    
    y_unlabeled = np.full(shape=y.shape, fill_value=MISSING_LABEL)
    batch_size = max(1, X_test.shape[0] // 10)  # Use max to avoid batch_size of 0
    st.write('batch size is', batch_size)
    y_unlabeled[:batch_size * 2] = y[:batch_size * 2]
    model.fit(X, y_unlabeled)

    # Evaluate classifier
    y_pred = model.predict(X_test)
    accuracy = metric(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")

    # Experiment loop
    for round_num in range(rounds):
        if R:
            selected_indices = strategy.query(X=X, y=y_unlabeled, batch_size=batch_size)
        else:
            selected_indices = strategy.query(X=X, y=y_unlabeled, clf=model, batch_size=batch_size)
        y_unlabeled[selected_indices] = y[selected_indices]

        # Train classifier with updated labeled data
        model.fit(X, y_unlabeled)

        # Evaluate classifier
        y_pred = model.predict(X_test)
        accuracy = metric(y_test, y_pred)
        st.write(f"Round {round_num + 1} - Accuracy: {accuracy}")

    # Final evaluation
    y_pred_final = model.predict(X_test)
    final_accuracy = metric(y_test, y_pred_final)
    st.markdown(f"**Final Accuracy: {final_accuracy}**")

def process_apple_quality_data(data):
    data = data.iloc[:-2]
    data['Acidity'] = data['Acidity'].astype(float)
    data['Quality'] = pd.get_dummies(data['Quality'], drop_first=True).astype(float)
    data.drop("A_id", axis=1, inplace=True)
    X, Y = data.drop("Quality", axis=1), data["Quality"]
    X, Y = X.to_numpy(), Y.to_numpy()
    X = (X - np.mean(X, axis=0)) / np.std(X)
    return X, Y

def process_diabetes_data(data):
    for i in ["gender", "smoking_history"]:
        new_cols = pd.get_dummies(data[i], drop_first=True)
        data = pd.concat([data, new_cols], axis=1)
        data.drop(i, axis=1, inplace=True)
    for i in data.columns:
        data[i] = data[i].astype(float)

    label_encoder = LabelEncoder()
    data["diabetes"] = label_encoder.fit_transform(data["diabetes"])

    X, Y = data.drop("diabetes", axis=1), data["diabetes"]
    X, Y = X.to_numpy(), Y.to_numpy()
    X = (X - np.mean(X, axis=0)) / np.std(X)
    return X, Y

master_random_state = np.random.RandomState(0)

def gen_seed(random_state: np.random.RandomState):
    return random_state.randint(0, 2**31)

def gen_random_state(random_state: np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))

def run_model(X, Y, qs_name, num_rounds):
    def create_query_strategy(name, random_state):
        return query_strategy_factory_functions[name](random_state)

    # Split the data
    X_test, X_train, y_test, y_train = train_test_split(X, Y, test_size=0.9, random_state=42, shuffle=True)

    # Create the query strategy
    qs = create_query_strategy(qs_name, gen_random_state(master_random_state))
    st.write(qs_name)

    clf = XGBClassifier()

    model = SklearnClassifier(
        clf,  # Use the XGBClassifier
        classes=np.unique(y_train),
        random_state=0
    )

    # Call run_active_learning with appropriate arguments
    if qs_name in ["RandomSampling", "CostEmbeddingAL"]:
        run_active_learning(X_train, y_train, X_test, y_test, model, qs, num_rounds, R=True, metric=f1_score)
    else:
        run_active_learning(X_train, y_train, X_test, y_test, model, qs, num_rounds, metric=f1_score)

    return model
def activate(selected_dataset, selected_strategy_name, num_rounds):
    if selected_dataset == "Apple Quality":
        data = pd.read_csv("apple_quality.csv")
        X, Y = process_apple_quality_data(data)
        # st.write(X.shape)
        # st.write(Y.shape)
    elif selected_dataset == "Diabetes":
        data = pd.read_csv("diabetes.csv")
        X, Y = process_diabetes_data(data)
    else:
        st.write("Please select a dataset.")
        return

    return run_model(X, Y, selected_strategy_name, num_rounds)

query_strategy_factory_functions = {
    'RandomSampling': lambda random_state: RandomSampling(random_state=gen_seed(random_state)),
    'UncertaintySampling (Entropy)': lambda random_state: UncertaintySampling(random_state=gen_seed(random_state), method='entropy'),
    'CostEmbeddingAL': lambda random_state: CostEmbeddingAL(random_state=gen_seed(random_state), classes=[0, 1]),
    'ProbabilisticAL': lambda random_state: ProbabilisticAL(random_state=gen_seed(random_state), metric='rbf')
}

# Streamlit app layout
st.title('Active Learning')

# Create two vertical groups
train_col, infer_col = st.columns(2)

# Training section
with train_col:
    st.subheader("Training")

    # Dataset selection
    selected_dataset = st.selectbox("Select the dataset:", options=["Apple Quality", "Diabetes"])

    num_rounds = st.number_input("Enter the number of rounds:", min_value=1, value=100, step=1)

    # Strategy selection
    selected_strategy_name = st.selectbox("Select the active learning strategy:", options=list(query_strategy_factory_functions.keys()))

    run_button = st.button('Run Model')

    if run_button:
        model = activate(selected_dataset, selected_strategy_name, num_rounds)
                # Inference section (after model training)
        if 'model' in locals():  # Check if model is trained
            with infer_col:
                st.subheader("Inference")

                input_data = None
                predict_button = st.button('Predict')
                # Input fields based on selected dataset
                if selected_dataset == "Apple Quality":
                    pass 
                elif selected_dataset == "Diabetes":
                    # Input fields for Diabetes dataset
                    gender = st.selectbox("Gender:", options=["Male", "Female", "Other"])
                    age = st.number_input("Age:", min_value=0, max_value=80, value=40)  # Set age range
                    hypertension = st.selectbox("Hypertension:", options=["No", "Yes"])
                    heart_disease = st.selectbox("Heart Disease:", options=["No", "Yes"])
                    smoking_history = st.selectbox("Smoking History:", options=["Not Current", "Former", "No Info", "Current", "Never", "Ever"])
                    bmi = st.number_input("BMI:", min_value=10.16, max_value=71.55, value=27.3)  # Set BMI range
                    hba1c_level = st.number_input("HbA1c Level:", min_value=3.5, max_value=9, value=5.53)  # Set HbA1c range
                    blood_glucose_level = st.number_input("Blood Glucose Level:", min_value=80, max_value=300, value=138)  # Set blood glucose range

                    
                    if predict_button:
                        # Encode categorical features
                        gender_encoded = 0  # Default to "Other"
                        if gender == "Male":
                            gender_encoded = 1
                        elif gender == "Female":
                            gender_encoded = 2

                        hypertension_encoded = 1 if hypertension == "Yes" else 0
                        heart_disease_encoded = 1 if heart_disease == "Yes" else 0

                        # Create input data
                        input_data = np.array([[0, gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
                                                smoking_history, bmi, hba1c_level, blood_glucose_level]])


                    # Make prediction and display
                    prediction = model.predict(input_data)
                    st.write("Prediction:", prediction[0])

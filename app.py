# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits 
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
from skactiveml.pool import UncertaintySampling, QueryByCommittee, RandomSampling, ProbabilisticAL, MonteCarloEER
from xgboost import XGBClassifier
from skactiveml.classifier import SklearnClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from skactiveml.utils import MISSING_LABEL
import warnings
warnings.filterwarnings("ignore")

def run_active_learning(X , y,X_test,y_test, clf , strategy, rounds,R=False):
    model = SklearnClassifier(
    GaussianProcessClassifier(random_state=0),
    classes=np.unique(y),
    random_state=0
    )
    
    y_unlabeled = np.full(shape=y.shape, fill_value=MISSING_LABEL)

    # Define batch size
    batch_size = X_test.shape[0] // 10
    y_unlabeled[:batch_size*2] = y[:batch_size*2]
    model.fit(X, y_unlabeled)

    # Evaluate classifier
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")

    # Experiment loop
    for round_num in range(rounds):
        
        if R:
            selected_indices = strategy.query(X=X, y=y_unlabeled,batch_size=batch_size)
        else:
            selected_indices = strategy.query(X=X, y=y_unlabeled, clf=model,batch_size=batch_size)
        y_unlabeled[selected_indices] = y[selected_indices]

        # Train classifier with updated labeled data
        model.fit(X, y_unlabeled)
        
        # Evaluate classifier
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Round {round_num + 1} - Accuracy: {accuracy}")

    # Final evaluation
    y_pred_final = model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    st.markdown(f"**Final Accuracy: {final_accuracy}**")


def process_data(data):
    data = data.iloc[:-2]
    data['Acidity'] = data['Acidity'].astype(float)
    data['Quality'] = pd.get_dummies(data['Quality'], drop_first=True).astype(float)
    data.drop("A_id", axis=1 , inplace=True)
    X , Y = data.drop("Quality", axis=1) ,data["Quality"]
    X,Y = X.to_numpy() ,Y.to_numpy()
    X = (X- np.mean(X,axis=0))/np.std(X)
    return X, Y

master_random_state = np.random.RandomState(0)

def gen_seed(random_state:np.random.RandomState):
    return random_state.randint(0, 2**31)

def gen_random_state(random_state:np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))

def run_model(X,Y,qs_name , num_rounds):
    def create_query_strategy(name, random_state):
        return query_strategy_factory_functions[name](random_state)
    
    # Split the data
    X_test, X_train, y_test, y_train = train_test_split(X, Y, test_size=0.9, random_state=42, shuffle=True)

    # Now create the query strategy here where you have access to random_state
    qs = create_query_strategy(qs_name, gen_random_state(master_random_state))
    st.write(qs_name)
    clf = XGBClassifier()
    
    # Depending on the query strategy name, call run_active_learning with the correct arguments
    if qs_name in ["RandomSampling","CostEmbeddingAL"] :
        run_active_learning(X_train, y_train, X_test, y_test, clf, qs, num_rounds, R=True)
    else:
        run_active_learning(X_train, y_train, X_test, y_test, clf, qs, num_rounds)



def activate(uploaded_file,selected_strategy_name , num_rounds):
    X, Y = process_data(uploaded_file)
    run_model(X,Y,selected_strategy_name , num_rounds)

query_strategy_factory_functions = {
    'RandomSampling': lambda random_state: RandomSampling(random_state=gen_seed(random_state)),
    'UncertaintySampling (Entropy)': lambda random_state: UncertaintySampling(random_state=gen_seed(random_state),method='entropy'),
    'CostEmbeddingAL': lambda random_state: CostEmbeddingAL(random_state=gen_seed(random_state),classes=[0, 1]),
    'ProbabilisticAL': lambda random_state: ProbabilisticAL(random_state=gen_seed(random_state), metric='rbf')
}

# Streamlit app layout
st.title('Model Prediction App')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"])
num_rounds = st.number_input("Enter the number of rounds:", min_value=1, value=100, step=1)

# Strategy selection
selected_strategy_name = st.selectbox("Select the active learning strategy:", options=list(query_strategy_factory_functions.keys()))

run_button = st.button('Run Model')

if run_button:
    if uploaded_file is None:
        st.write("Please upload a CSV file to get started.")
    else:
        # Read the uploaded CSV
        input_df = pd.read_csv(uploaded_file)

        selected_strategy = query_strategy_factory_functions[selected_strategy_name]

        # Assuming 'predict' is your model's prediction function
        prediction = activate(input_df ,selected_strategy_name , num_rounds)

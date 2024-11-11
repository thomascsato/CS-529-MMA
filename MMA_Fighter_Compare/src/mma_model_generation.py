import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# 1. Data Preparation
def prepare_data(df):
    # Create features and target variable
    pre_features_numeric = ["r_wins_agg",
       "r_losses_agg", "r_height_agg", "r_weight_agg", "r_reach_agg", "r_age_agg",
       "r_SLpM_agg", "r_sig_str_acc_agg", "r_SApM_agg", "r_str_def_agg",
       "r_td_avg_agg", "r_td_acc_agg", "r_td_def_agg", "r_sub_avg_agg", "b_wins_agg",
       "b_losses_agg", "b_height_agg", "b_weight_agg", "b_reach_agg", "b_age_agg",
       "b_SLpM_agg", "b_sig_str_acc_agg", "b_SApM_agg", "b_str_def_agg",
       "b_td_avg_agg", "b_td_acc_agg", "b_td_def_agg", "b_sub_avg_agg"]
    pre_features_categorical = ["weight_class", "is_title_bout", "gender",
       "r_stance_agg", "b_stance_agg"]
    
    all_features_numeric = pre_features_numeric + ["finish_round", "total_rounds",
       "time_sec", "r_kd", "r_sig_str", "r_sig_str_att", "r_sig_str_acc", "r_str",
       "r_str_att", "r_str_acc", "r_td", "r_td_att", "r_td_acc",
       "r_sub_att", "r_rev", "r_ctrl_sec", "b_kd", "b_sig_str",
       "b_sig_str_att", "b_sig_str_acc", "b_str", "b_str_att", "b_str_acc",
       "b_td", "b_td_att", "b_td_acc", "b_sub_att", "b_rev", "b_ctrl_sec"]
    all_features_categorical = pre_features_categorical + ["method"]

    pre_features = pre_features_numeric + pre_features_categorical
    all_features = all_features_numeric + all_features_categorical
    
    X_pre = df[pre_features]
    X_all = df[all_features]
    y = df["winner"]  # Assuming 1 for r_fighter win, 0 for b_fighter win
    
    return X_pre, X_all, y, pre_features_numeric, pre_features_categorical, all_features_numeric, all_features_categorical

# 2. Model Training
def train_model(X_all, y, all_features_numeric, all_features_categorical):
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_features_numeric),
            ('cat', categorical_transformer, all_features_categorical)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    
    return model

#3. save model
def save_model(model):
    joblib.dump(model, 'mma_model.joblib')

if __name__ == "__main__":
    # Load your fight data
    df = pd.read_csv('MMA_Fighter_Compare\\data\\all_fights_final.csv')
    
    # Prepare data
    X_pre_fight, X_all, y, pre_fight_numeric, pre_fight_categorical, all_numeric, all_categorical = prepare_data(df)
    
    # Train model on all features
    model = train_model(X_all, y, all_numeric, all_categorical)
    
    # Save model
    save_model(model)
    
    print("Model saved successfully.")
    
    # Example of how to use the model for prediction (using only pre-fight features)
    def predict_fight(pre_fight_data, model, pre_fight_numeric, pre_fight_categorical, all_numeric, all_categorical):
        # Create a DataFrame with pre-fight data
        pre_fight_df = pd.DataFrame([pre_fight_data], columns=pre_fight_numeric + pre_fight_categorical)
        
        # Create a full feature set with placeholder values for post-fight features
        full_feature_set = pre_fight_df.copy()
        for feature in all_numeric:
            if feature not in pre_fight_numeric:
                full_feature_set[feature] = 0  # or some other placeholder value
        for feature in all_categorical:
            if feature not in pre_fight_categorical:
                full_feature_set[feature] = 'unknown'  # or some other placeholder value
        
        # Ensure columns are in the correct order
        full_feature_set = full_feature_set[all_numeric + all_categorical]
        
        # Make prediction
        return model.predict_proba(full_feature_set)[0]
    
    # Example usage (you would replace this with actual fighter data)
    example_pre_fight_data = [0] * len(pre_fight_numeric) + ['Lightweight', False, 'Male', 'Orthodox', 'Southpaw']
    win_probabilities = predict_fight(example_pre_fight_data, model, pre_fight_numeric, pre_fight_categorical, all_numeric, all_categorical)
    print(f"Fighter 1 win probability: {win_probabilities[1]:.2f}")
    print(f"Fighter 2 win probability: {win_probabilities[0]:.2f}")
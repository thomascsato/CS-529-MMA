import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_data(df):

    # Create pre-fight features for both fighters
    pre_features_numeric = ["r_wins_agg",
        "r_losses_agg", "r_height_agg", "r_weight_agg", "r_reach_agg", "r_age_agg",
        "r_SLpM_agg", "r_sig_str_acc_agg", "r_SApM_agg", "r_str_def_agg",
        "r_td_avg_agg", "r_td_acc_agg", "r_td_def_agg", "r_sub_avg_agg", "b_wins_agg",
        "b_losses_agg", "b_height_agg", "b_weight_agg", "b_reach_agg", "b_age_agg",
        "b_SLpM_agg", "b_sig_str_acc_agg", "b_SApM_agg", "b_str_def_agg",
        "b_td_avg_agg", "b_td_acc_agg", "b_td_def_agg", "b_sub_avg_agg"]
    pre_features_categorical = ["weight_class", "is_title_bout", "gender", "r_stance_agg", "b_stance_agg"]

    # Create post-fight features for both fighters
    post_features_numeric = ["finish_round", "total_rounds", "time_sec", 
        "r_kd", "r_sig_str", "r_sig_str_att", "r_sig_str_acc", "r_str",
        "r_str_att", "r_str_acc", "r_td", "r_td_att", "r_td_acc",
        "r_sub_att", "r_rev", "r_ctrl_sec", "b_kd", "b_sig_str",
        "b_sig_str_att", "b_sig_str_acc", "b_str", "b_str_att", "b_str_acc",
        "b_td", "b_td_att", "b_td_acc", "b_sub_att", "b_rev", "b_ctrl_sec"]
    post_features_categorical = ["method"]

    # Combining pre fight features with post fight features
    all_features_numeric = pre_features_numeric + post_features_numeric
    all_features_categorical = pre_features_categorical + post_features_categorical

    # Combining pre-fight features and post-fight features
    all_pre_features = pre_features_numeric + pre_features_categorical
    all_post_features = post_features_numeric + post_features_categorical
    all_features = all_features_numeric + all_features_categorical
    
    # Splitting predictors and targets
    X_pre = df[all_pre_features]
    X_all = df[all_features]
    y_post = df[all_post_features]
    y_win = df["winner"]  # Assuming 1 for r_fighter win, 0 for b_fighter win
    
    return X_pre, X_all, y_post, y_win, pre_features_numeric, pre_features_categorical, all_features_numeric, all_features_categorical

def train_post_fight_model(X_pre, y_post_fight, all_features_numeric, all_features_categorical):
    """This function aims to predict post-fight statistics from two fighters.
    This will then be fed into a second model that predicts win probabilities.
    """

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y_post_fight, test_size=0.2, random_state=42)

    # Transform numeric variables with scaling and imputation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Transform categorical variables with one hot encoding and imputation
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])

    # Use transformers to preprocess columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_features_numeric),
            ('cat', categorical_transformer, all_features_categorical)
        ])
    
    # Multi-output regressor handles multiple target variables
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('multi_output_regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
    ])
    
    # Fit the pipeline
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    return model

def train_win_prob_model(X_all, y, all_features_numeric, all_features_categorical):
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

def save_model(model):
    joblib.dump(model, 'mma_model.joblib')

if __name__ == "__main__":
    
    # Load your fight data
    df = pd.read_csv('MMA_Fighter_Compare\\data\\all_fights_final.csv')
    
    # Prepare data
    X_pre_fight, X_all, y_winner, pre_fight_numeric, pre_fight_categorical, all_numeric, all_categorical = prepare_data(df)
    
    # Train post-fight statistics prediction model (you need to define your target variable for post-fight stats)
    # For example: Assuming you have a target variable like 'total_significant_strikes' for fighter A.
    
    # Create your target variable for post-fight stats (this is just an example; adjust according to your dataset)
    y_post_fight_stats = df['total_significant_strikes']  # Replace with actual post-fight stat you want to predict
    
    post_fight_model = train_post_fight_model(X_pre_fight, y_post_fight_stats)


    # Train win prediction model on all features including predicted post-fight stats later on.
    win_prediction_model = train_win_prob_model(X_all, y_winner)

    # Save models
    save_model(post_fight_model)
    save_model(win_prediction_model)

    print("Models saved successfully.")

    # Example of how to use the models for prediction:
    def predict_fight(fighter1_data, fighter2_data):
        expected_length = len(pre_fight_numeric)
        if len(fighter1_data) != expected_length or len(fighter2_data) != expected_length:
            print(f"Expected feature length: {expected_length}")
            print(f"Fighter 1 data length: {len(fighter1_data)}")
            print(f"Fighter 2 data length: {len(fighter2_data)}")
            raise ValueError("Fighter data must match expected feature length.")

        # Combine fighter data into a single list with additional static features
        pre_fight_data = fighter1_data + fighter2_data + ['Lightweight', False, 'Male']

        # Create a DataFrame with combined fighter data
        pre_fight_df = pd.DataFrame([pre_fight_data], columns=pre_fight_numeric + pre_fight_categorical)

        # Predict post-fight statistics using the first model
        predicted_post_stats = post_fight_model.predict(pre_fight_df)

        # Combine predicted stats with original features for win prediction input
        combined_input_for_win_prediction = np.concatenate((fighter1_data + fighter2_data,
                                                            predicted_post_stats.tolist(),
                                                            ['Lightweight', False]))  # Add static features if needed

        combined_input_df = pd.DataFrame([combined_input_for_win_prediction],
                                            columns=all_numeric + all_categorical)

        # Make win prediction using the second model
        win_probabilities = win_prediction_model.predict_proba(combined_input_df)[0]
        
        return win_probabilities[1], win_probabilities[0]  # Return probabilities for fighter1 and fighter2

    # Example usage (ensure this matches your expected input structure)
    fighter1_data = [10, 2, 180, 77, 72, 30,
                    4.5 ,0.6 ,3.2 ,0.7,
                    2.1 ,0.5 ,0.6 ,0.2]  
                    
    fighter2_data = [8 ,3 ,178 ,77 ,70 ,28,
                    3.8 ,0.55 ,2.9 ,0.65,
                    1.8 ,0.45 ,0.7 ,0.3]  

    fighter1_prob, fighter2_prob = predict_fight(fighter1_data,
                                                    fighter2_data)

    print(f"Fighter 1 win probability: {fighter1_prob:.2f}")
    print(f"Fighter 2 win probability: {fighter2_prob:.2f}")
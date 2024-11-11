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

    global pre_features_numeric, pre_features_categorical, post_features_numeric, post_features_categorical

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
    
    return X_pre, X_all, y_post, y_win, all_features_numeric, all_features_categorical

def train_post_fight_model(X_pre, y_post, pre_features_numeric, pre_features_categorical):
    """This function aims to predict post-fight statistics from two fighters.
    This will then be fed into a second model that predicts win probabilities.
    """

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y_post, test_size=0.2, random_state=42)

    # Reset index (Important for rejoining y_train)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Imputing target values here
    y_train_numeric = y_train.select_dtypes(include=['float64', 'int64'])
    y_train_categorical = y_train.select_dtypes(include=['object', 'category'])

    imputer = SimpleImputer(strategy='mean')
    y_train_imputed = pd.DataFrame(imputer.fit_transform(y_train_numeric), columns=y_train_numeric.columns)

    # Concatenate numeric and categorical columns to form the full y_train_imputed DataFrame
    y_train_imputed = pd.concat([y_train_imputed, y_train_categorical], axis=1)

    # Ensure the order of columns matches the original y_train
    y_train_imputed = y_train_imputed[y_train.columns]
    
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
            # Transforming only on X_train columns (pre-fight features)
            ('num', numeric_transformer, pre_features_numeric),
            ('cat', categorical_transformer, pre_features_categorical)
        ])
    
    # Multi-output regressor handles multiple target variables
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('multi_output_regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
    ])

    # Fit the pipeline
    print("Fitting model.")
    model.fit(X_train, y_train_imputed)
    print("Model fit successfully.")
    
    # Evaluate model
    print("Evaluating model.")
    y_pred = model.predict(X_test)
    print("Model evaluated successfully.")
    
    return model, y_pred

def train_win_prob_model(X_all, y_win, all_features_numeric, all_features_categorical):
    """This model aims to predict win probabilities based on predicted post-fight stats."""

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_win, test_size=0.2, random_state=42)

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
    X_pre, X_all, y_post, y_win, all_numeric, all_categorical = prepare_data(df)
    
    # Train first model to predict 
    post_fight_model, prediction = train_post_fight_model(X_pre, y_post, pre_features_numeric, pre_features_categorical)
    print(prediction)

import os
import io
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def model_fn(model_dir):
    """Load the model from the model directory."""
    model_path = os.path.join(model_dir, 'win_pred_model.joblib')
    return joblib.load(model_path)


def predict_fn(input_data, model):
    prob = model.predict_proba(input_data)
    return prob


def input_fn(input_data, content_type):
    """Converts input data to a Pandas DataFrame."""

    if content_type.lower() == 'text/csv':

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

        column_names = pre_features_numeric + pre_features_categorical + post_features_numeric

        try:
            # Convert CSV input to Pandas DataFrame
            df = pd.read_csv(io.StringIO(input_data), header=None, names=column_names)

            # Validate that the DataFrame has all expected columns
            missing_columns = [col for col in column_names if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in input data: {missing_columns}")

            return df

        except Exception as e:
            raise ValueError(f"Error processing input data: {e}")

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def duplicate_and_switch(df):
    """This function will double the dataset, switching the red and blue fighter in order
    to not fit the model based on order of fighter in the data."""

    # Create a copy of the original DataFrame
    df_switched = df.copy()

    # Define columns that need to be switched
    columns_to_switch = [
        "wins_agg", "losses_agg", "height_agg", "weight_agg", "reach_agg",
        "SLpM_agg", "sig_str_acc_agg", "SApM_agg", "str_def_agg",
        "td_avg_agg", "td_acc_agg", "td_def_agg", "sub_avg_agg", 
        "age_agg", "kd", "sig_str", "sig_str_att", "sig_str_acc",
        "str_att", "str_acc", "td", "td_att", "td_acc", "str",
        "sub_att", "rev", "ctrl_sec", "stance_agg"
    ]

    # Swap the corresponding 'r_' and 'b_' columns
    for col in columns_to_switch:
        df_switched['b_' + col] = df['r_' + col]  # Assign red fighter stats to blue fighter
        df_switched['r_' + col] = df['b_' + col]  # Assign blue fighter stats to red fighter

    # Adjust winner column based on new roles
    df_switched['winner'] = df['winner'].apply(lambda x: 1 if x == 0 else 0)  # Assuming 1 for red win, 0 for blue win

    return pd.concat([df, df_switched], ignore_index=True)


def prepare_data(df):

    global pre_features_numeric, pre_features_categorical, post_features_numeric

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

    # Combining pre fight features with post fight features
    all_features_numeric = pre_features_numeric + post_features_numeric
    all_features_categorical = pre_features_categorical

    # Combining pre-fight features and post-fight features
    all_pre_features = pre_features_numeric + pre_features_categorical
    all_post_features = post_features_numeric
    all_features = pre_features_numeric + pre_features_categorical + post_features_numeric
    
    # Splitting predictors and targets
    X_pre = df[all_pre_features]
    X_all = df[all_features]
    y_post = df[all_post_features]
    y_win = df["winner"]  # Assuming 1 for r_fighter win, 0 for b_fighter win
    
    return X_pre, X_all, y_post, y_win, all_features_numeric, all_features_categorical


def train_win_prob_model(X_all, y_win, all_features_numeric, all_features_categorical):
    """This model aims to predict win probabilities based on predicted post-fight stats."""

    # Training on all data available
    X_train = X_all
    y_train = y_win

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_features_numeric),
            ('cat', categorical_transformer, all_features_categorical)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, verbose=2))
    ])
    
    # Fit the pipeline
    model.fit(X_train, y_train)

    joblib.dump(model, 'MMA_Fighter_Compare\\models\\win_pred_model_1.joblib')

    return model


def main():
    train_data_path = os.environ['SM_CHANNEL_TRAIN']
    train_data = pd.read_csv(os.path.join(train_data_path, 'all_fights_final.csv'))

    # Duplicate values and concatenate
    df_full = duplicate_and_switch(train_data)

    # Prepare data
    X_pre, X_all, y_post, y_win, all_numeric, all_categorical = prepare_data(df_full)

    # Train win probability prediction model
    win_prob_model = train_win_prob_model(X_all, y_win, all_numeric, all_categorical)

    # Save model to model directory
    model_output_path = os.environ['SM_MODEL_DIR']
    joblib.dump(win_prob_model, os.path.join(model_output_path, 'win_pred_model.joblib'))


if __name__ == "__main__":
    main()
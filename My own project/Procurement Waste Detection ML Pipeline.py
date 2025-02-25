import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. Data Loading & Exploration
# ========================

def load_and_explore_data(file_path):
    """
    Load procurement data and perform initial exploration
    """
    print("Loading and exploring procurement data...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    return df

# ========================
# 2. Feature Engineering
# ========================

def engineer_features(df):
    """
    Create features that might be predictive of wasteful spending
    """
    print("\nEngineering features...")
    
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Example features that might indicate wasteful spending
    
    # 1. Price deviation from category average
    # Group by category and calculate mean price
    category_avg_price = df_processed.groupby('item_category')['price'].transform('mean')
    df_processed['price_deviation'] = (df_processed['price'] - category_avg_price) / category_avg_price
    
    # 2. Rush orders (if order_date and required_date fields exist)
    if 'order_date' in df_processed.columns and 'required_date' in df_processed.columns:
        df_processed['order_date'] = pd.to_datetime(df_processed['order_date'])
        df_processed['required_date'] = pd.to_datetime(df_processed['required_date'])
        df_processed['lead_time'] = (df_processed['required_date'] - df_processed['order_date']).dt.days
        df_processed['is_rush_order'] = df_processed['lead_time'] < 5  # Define rush orders as < 5 days
    
    # 3. End of budget period purchases (if date field exists)
    if 'order_date' in df_processed.columns:
        df_processed['month'] = df_processed['order_date'].dt.month
        df_processed['is_end_of_quarter'] = df_processed['month'].isin([3, 6, 9, 12])
    
    # 4. Unusual quantities
    quantity_avg = df_processed.groupby('item_category')['quantity'].transform('mean')
    quantity_std = df_processed.groupby('item_category')['quantity'].transform('std')
    df_processed['quantity_zscore'] = (df_processed['quantity'] - quantity_avg) / quantity_std
    
    # 5. Weekend orders (if date field exists)
    if 'order_date' in df_processed.columns:
        df_processed['is_weekend'] = df_processed['order_date'].dt.dayofweek >= 5
    
    # 6. Unusual supplier choices
    # Count how often each supplier is used for each category
    supplier_category_counts = df_processed.groupby(['supplier_id', 'item_category']).size().reset_index(name='counts')
    common_suppliers = supplier_category_counts.groupby('item_category')['supplier_id'].apply(list).to_dict()
    
    # Mark orders with uncommon suppliers
    df_processed['is_uncommon_supplier'] = False
    for idx, row in df_processed.iterrows():
        category = row['item_category']
        supplier = row['supplier_id']
        if category in common_suppliers:
            category_suppliers = common_suppliers[category]
            # If this supplier is not among the top suppliers for this category
            if supplier not in category_suppliers[:3]:  # Assuming top 3 are common
                df_processed.at[idx, 'is_uncommon_supplier'] = True
    
    print("Engineered features preview:")
    new_columns = [col for col in df_processed.columns if col not in df.columns]
    print(df_processed[new_columns].head())
    
    return df_processed

# ========================
# 3. Data Preprocessing
# ========================

def preprocess_data(df, target_column='is_wasteful'):
    """
    Prepare data for machine learning
    """
    print("\nPreprocessing data for machine learning...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

# ========================
# 4. Model Training and Evaluation
# ========================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train multiple models and evaluate their performance
    """
    print("\nTraining and evaluating models...")
    
    # Define models to try
    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ]),
        
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
    }
    
    # Results storage
    model_results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Cross-validation score
        cv_score = cross_val_score(model, pd.concat([X_train, X_test]), 
                                  pd.concat([y_train, y_test]), 
                                  cv=5, scoring='roc_auc').mean()
        
        # Store results
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'cv_score': cv_score
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"5-Fold CV ROC AUC: {cv_score:.4f}")
        print("Classification Report:")
        print(report)
    
    # Find best model based on ROC AUC
    best_model_name = max(model_results, key=lambda x: model_results[x]['roc_auc'])
    print(f"\nBest model: {best_model_name} with ROC AUC of {model_results[best_model_name]['roc_auc']:.4f}")
    
    return model_results, best_model_name

# ========================
# 5. Feature Importance Analysis
# ========================

def analyze_feature_importance(model_results, best_model_name, X_train, preprocessor):
    """
    Analyze and visualize feature importance for the best model
    """
    print("\nAnalyzing feature importance...")
    
    best_model = model_results[best_model_name]['model']
    
    # For tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        # Get the trained model from the pipeline
        model = best_model.named_steps['classifier']
        
        # Get feature names after preprocessing
        if hasattr(preprocessor, 'transformers_'):
            # Get column names from the preprocessor
            cat_cols = preprocessor.transformers_[1][2]  # Categorical columns
            cat_transformer = preprocessor.transformers_[1][1]  # Categorical transformer
            
            # Get the one-hot encoder from the categorical transformer
            if 'onehot' in cat_transformer.named_steps:
                onehot = cat_transformer.named_steps['onehot']
                if hasattr(onehot, 'get_feature_names_out'):
                    cat_features = onehot.get_feature_names_out(cat_cols)
                else:
                    cat_features = [f"{col}_{val}" for col in cat_cols for val in onehot.categories_[cat_cols.index(col)]]
            else:
                cat_features = cat_cols
                
            num_cols = preprocessor.transformers_[0][2]  # Numerical columns
            all_features = list(num_cols) + list(cat_features)
        else:
            all_features = [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Extract feature importances
        importances = model.feature_importances_
        
        # Match feature importances with feature names
        if len(importances) == len(all_features):
            feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display top 15 features
            top_features = feature_importance.head(15)
            print("\nTop 15 features:")
            print(top_features)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Top 15 Feature Importances for {best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
            print("Feature importance plot saved as 'feature_importance.png'")
            
            return feature_importance
    
    # For logistic regression
    elif best_model_name == 'Logistic Regression':
        print("Feature importance analysis for Logistic Regression is not implemented in this example.")
    
    return None

# ========================
# 6. Model Optimization
# ========================

def optimize_best_model(best_model_name, X_train, y_train, X_test, y_test, preprocessor):
    """
    Optimize the best model using hyperparameter tuning
    """
    print(f"\nOptimizing {best_model_name} with hyperparameter tuning...")
    
    param_grid = {}
    
    if best_model_name == 'Logistic Regression':
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    
    elif best_model_name == 'Random Forest':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5]
        }
    
    elif best_model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    # Create base model pipeline
    base_model = None
    if best_model_name == 'Logistic Regression':
        base_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
    elif best_model_name == 'Random Forest':
        base_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])
    elif best_model_name == 'Gradient Boosting':
        base_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
    elif best_model_name == 'XGBoost':
        base_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
    
    # Perform grid search
    if base_model is not None and param_grid:
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        # Evaluate optimized model
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = best_model.score(X_test, y_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"\nOptimized {best_model_name} Results:")
        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Classification Report:")
        print(report)
        
        return best_model, best_params
    
    print("Optimization not applicable for the selected model.")
    return None, None

# ========================
# 7. Save Model
# ========================

def save_model(model, file_path='wasteful_procurement_model.pkl'):
    """
    Save the trained model to a file
    """
    print(f"\nSaving model to {file_path}...")
    joblib.dump(model, file_path)
    print("Model saved successfully.")

# ========================
# 8. Create Wasteful Purchase Detection Function
# ========================

def detect_wasteful_purchases(model, new_data, threshold=0.5):
    """
    Detect potentially wasteful purchases in new procurement data
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    new_data : pandas.DataFrame
        New procurement data to evaluate
    threshold : float, default=0.5
        Probability threshold for classifying as wasteful
        
    Returns:
    --------
    pandas.DataFrame
        Original data with added wasteful prediction and probability columns
    """
    print("\nDetecting potentially wasteful purchases...")
    
    # Make a copy of the input data
    result_df = new_data.copy()
    
    # Make predictions
    probabilities = model.predict_proba(new_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    # Add predictions to the dataframe
    result_df['wasteful_probability'] = probabilities
    result_df['is_wasteful_predicted'] = predictions
    
    # Sort by probability (highest first)
    result_df = result_df.sort_values('wasteful_probability', ascending=False)
    
    # Summary statistics
    total_purchases = len(result_df)
    wasteful_purchases = sum(predictions)
    wasteful_percentage = (wasteful_purchases / total_purchases) * 100
    
    print(f"Total purchases evaluated: {total_purchases}")
    print(f"Potentially wasteful purchases detected: {wasteful_purchases} ({wasteful_percentage:.2f}%)")
    
    return result_df

# ========================
# 9. Main Function
# ========================

def main():
    """
    Main function to run the entire pipeline
    """
    print("Starting wasteful procurement detection pipeline...")
    
    # Sample usage - in practice, replace with your actual data file
    data_file = "procurement_data.csv"
    
    try:
        # Load and explore data
        df = load_and_explore_data(data_file)
        
        # Engineer features
        df_processed = engineer_features(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df_processed)
        
        # Train and evaluate models
        model_results, best_model_name = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(model_results, best_model_name, X_train, preprocessor)
        
        # Optimize the best model
        optimized_model, best_params = optimize_best_model(best_model_name, X_train, y_train, X_test, y_test, preprocessor)
        
        # Save the optimized model
        if optimized_model is not None:
            save_model(optimized_model)
        else:
            save_model(model_results[best_model_name]['model'])
        
        print("\nWasteful procurement detection pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found. Please provide a valid file path.")
        print("This script expects a CSV file with procurement data including features like:")
        print("- item_category: Category of purchased item")
        print("- price: Cost of the item")
        print("- quantity: Number of items purchased")
        print("- supplier_id: ID of the supplier")
        print("- order_date: Date when order was placed (optional)")
        print("- required_date: Date when item is needed (optional)")
        print("- is_wasteful: Target variable indicating wasteful purchase (1/0)")
        
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

# Example of how to use the model for new data
def example_usage():
    """
    Example of how to use the trained model
    """
    # Load the trained model
    model = joblib.load('wasteful_procurement_model.pkl')
    
    # Load new procurement data
    new_data = pd.read_csv('new_procurement_data.csv')
    
    # Engineer features for new data (same as in training)
    new_data_processed = engineer_features(new_data)
    
    # Drop the target column if it exists in the new data
    if 'is_wasteful' in new_data_processed.columns:
        new_data_processed = new_data_processed.drop(columns=['is_wasteful'])
    
    # Detect wasteful purchases
    results = detect_wasteful_purchases(model, new_data_processed, threshold=0.7)
    
    # Save the results
    results.to_csv('wasteful_purchase_predictions.csv', index=False)
    
    # Display top 10 most likely wasteful purchases
    print("\nTop 10 most likely wasteful purchases:")
    print(results.head(10)[['item_category', 'price', 'quantity', 'wasteful_probability']])

if __name__ == "__main__":
    main()
    # Uncomment to run example usage
    # example_usage()

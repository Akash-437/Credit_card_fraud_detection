import os
import sys
import pickle
import pandas as pd
from colorama import init, Fore, Style, Back

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Initialize Colorama for cross-platform colored terminal output
init(autoreset=True)

# Constants
DATA_FILE = 'creditcard.csv'
MODEL_FILE = 'best_fraud_model.pkl'
SCALER_FILE = 'scaler.pkl'

def print_header(title):
    """Prints a formatted header."""
    print(Fore.CYAN + "\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + f"{title.center(70)}")
    print(Fore.CYAN + "="*70 + "\n")

def analyze_dataset():
    """Module 1: Dataset Analysis Output"""
    print_header("DATASET ANALYSIS")

    if not os.path.exists(DATA_FILE):
        print(Fore.RED + f"Error: Dataset '{DATA_FILE}' not found in the current directory.")
        return

    print(Fore.YELLOW + "Loading dataset... Please wait.\n")
    try:
        df = pd.read_csv(DATA_FILE)

        # Clean dataset by dropping any rows with missing values (NaN)
        df = df.dropna()

        total_transactions = len(df)
        legit_transactions = len(df[df['Class'] == 0])
        fraud_transactions = len(df[df['Class'] == 1])

        if total_transactions == 0:
            print(Fore.RED + "Error: Dataset is empty after removing missing values.")
            return

        fraud_percentage = (fraud_transactions / total_transactions) * 100
        num_features = df.shape[1] - 1 # Excluding target

        print(Fore.GREEN + f"Total transactions      : {total_transactions:,}")
        print(Fore.GREEN + f"Legitimate transactions : {legit_transactions:,}")
        print(Fore.RED + f"Fraudulent transactions : {fraud_transactions:,}")
        print(Fore.YELLOW + f"Percentage of fraud     : {fraud_percentage:.3f}%")
        print(Fore.GREEN + f"Number of features      : {num_features}\n")

        print(Style.BRIGHT + "Class Imbalance Statistics:")
        if fraud_transactions > 0:
            print(f"Ratio of Legit to Fraud : {legit_transactions//fraud_transactions}:1")
        else:
            print("Ratio of Legit to Fraud : No fraud cases found.")

    except Exception as e:
        print(Fore.RED + f"An error occurred while analyzing: {e}")

def train_models():
    """Module 2: Model Training and Evaluation Output"""
    print_header("MODEL TRAINING & EVALUATION")

    if not os.path.exists(DATA_FILE):
        print(Fore.RED + f"Error: Dataset '{DATA_FILE}' not found.")
        return

    try:
        # 1. Preprocessing
        print(Fore.YELLOW + "[1/5] Loading and preprocessing data...")
        df = pd.read_csv(DATA_FILE)

        # Clean dataset by dropping any rows with missing values (NaN)
        df = df.dropna()

        # Drop Time as it's not universally useful without feature engineering
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)

        X = df.drop('Class', axis=1)
        y = df['Class']

        # Scale Amount feature
        print(Fore.YELLOW + "[2/5] Scaling numerical features...")
        scaler = StandardScaler()
        if 'Amount' in X.columns:
            X['Amount'] = scaler.fit_transform(X[['Amount']])

        # 2. Train-Test Split
        print(Fore.YELLOW + "[3/5] Splitting data and applying SMOTE...")

        # Check if we have enough samples for SMOTE and train_test_split
        if len(y.unique()) < 2:
            print(Fore.RED + "Error: Only one class present in the dataset. Cannot train models.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 3. Handle Class Imbalance
        # Adjust SMOTE k_neighbors if fraud cases are extremely low
        min_fraud_cases = min(y_train.value_counts())
        k_neighbors = min(5, min_fraud_cases - 1)

        if k_neighbors > 0:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            print(Fore.RED + "Not enough fraud cases to apply SMOTE. Training on raw imbalanced data...")
            X_train_resampled, y_train_resampled = X_train, y_train

        # 4. Train Models
        print(Fore.YELLOW + "[4/5] Training models (This may take a few minutes)...\n")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        }

        results = {}
        best_f1 = 0
        best_model_name = ""
        best_model = None

        for name, model in models.items():
            print(Fore.CYAN + f"Training {name}...")
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'roc': roc, 'cm': cm, 'model': model}

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = model

        # Fallback if F1 is 0 for all models
        if best_model is None:
            best_model_name = "Random Forest"
            best_model = models["Random Forest"]

        # 5. Display Results
        print(Fore.YELLOW + "\n[5/5] Evaluation Complete. Generating Report...\n")

        print(Style.BRIGHT + "Confusion Matrices:")
        for name, res in results.items():
            print(f"\n{name}:")
            print(res['cm'])

        print("\n" + "="*70)
        print(Style.BRIGHT + f"{'Model':<22} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<8}")
        print("-" * 70)

        for name, res in results.items():
            print(f"{name:<22} {res['acc']:<10.4f} {res['prec']:<11.4f} {res['rec']:<8.4f} {res['f1']:<8.4f}")
        print("="*70 + "\n")

        print(Fore.GREEN + Style.BRIGHT + f"Best Model: {best_model_name} (F1-Score: {best_f1:.4f})\n")

        # Save Best Model and Scaler
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({'model': best_model, 'model_name': best_model_name}, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)

        print(Fore.CYAN + f"Saved best model and scaler to disk.")

    except Exception as e:
        print(Fore.RED + f"An error occurred during training: {e}")

def get_risk_level(fraud_prob):
    """Calculates risk level and visual bar based on probability."""
    prob = fraud_prob * 100
    if prob < 25:
        return Fore.GREEN + "████░░░░░░ LOW", "Low"
    elif prob < 50:
        return Fore.YELLOW + "███████░░░ MEDIUM", "Medium"
    elif prob < 75:
        return Fore.RED + "█████████░ HIGH", "High"
    else:
        return Fore.RED + Style.BRIGHT + "██████████ CRITICAL", "Critical"

def predict_transaction():
    """Module 3: Fraud Prediction Output"""
    print_header("FRAUD PREDICTION SYSTEM")

    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print(Fore.RED + "Error: Trained model or scaler not found.")
        print("Please run 'Train Models' from the main menu first.")
        return

    try:
        with open(MODEL_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            model_name = saved_data['model_name']

        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)

        print("How would you like to input the transaction details?")
        print("1. Paste a comma-separated row (29 features: V1-V28, Amount)")
        print("2. Enter manually")

        choice = input("\nEnter choice (1/2): ").strip()

        features = []
        amount = 0.0

        if choice == '1':
            row_input = input("\nPaste the 29 comma-separated values (V1-V28, Amount):\n")
            values = [float(x.strip()) for x in row_input.split(',')]
            if len(values) != 29:
                print(Fore.RED + f"Error: Expected 29 features, got {len(values)}.")
                return
            features = values[:-1]
            amount = values[-1]

        elif choice == '2':
            print("\nEntering 28 PCA features (V1-V28) and Amount.")
            for i in range(1, 29):
                val = float(input(f"Enter V{i}: "))
                features.append(val)
            amount = float(input("Enter Transaction Amount ($): "))
        else:
            print(Fore.RED + "Invalid choice.")
            return

        # Prepare for prediction
        input_df = pd.DataFrame([features + [amount]], columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
        input_df['Amount'] = scaler.transform(input_df[['Amount']])

        # Predict
        fraud_prob = model.predict_proba(input_df)[0][1]
        legit_prob = 1 - fraud_prob
        is_fraud = fraud_prob >= 0.5

        risk_bar, risk_text = get_risk_level(fraud_prob)

        # Output formatting
        print("\n" + Fore.CYAN + "══════════════════════════════════════════")

        if is_fraud:
            print(Back.RED + Fore.WHITE + Style.BRIGHT + " 🚨 FRAUDULENT TRANSACTION ".center(42))
        else:
            print(Back.GREEN + Fore.WHITE + Style.BRIGHT + " ✅ LEGITIMATE TRANSACTION ".center(42))

        print(Fore.CYAN + "══════════════════════════════════════════\n")

        print(f"Fraud probability : {fraud_prob*100:.2f}%")
        print(f"Legit probability : {legit_prob*100:.2f}%")
        print(f"Transaction amount: ${amount:.2f}")
        print(f"Model             : {model_name}\n")
        print(f"Risk: {risk_bar}")

        print(Fore.CYAN + "══════════════════════════════════════════")

    except ValueError:
        print(Fore.RED + "Error: Invalid numerical input. Ensure there are no strings in your data.")
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}")

def main():
    """Main menu loop."""
    while True:
        print_header("CREDIT CARD FRAUD DETECTION SYSTEM")
        print("1. Analyze Dataset")
        print("2. Train Models & Evaluate")
        print("3. Predict Transaction")
        print("4. Exit")

        choice = input(Fore.YELLOW + "\nEnter your choice [1-4]: " + Style.RESET_ALL).strip()

        if choice == '1':
            analyze_dataset()
        elif choice == '2':
            train_models()
        elif choice == '3':
            predict_transaction()
        elif choice == '4':
            print(Fore.GREEN + "Exiting system. Goodbye!")
            sys.exit(0)
        else:
            print(Fore.RED + "Invalid choice. Please enter a number between 1 and 4.")

        input("\nPress Enter to continue...")
        # Clear screen based on OS
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()
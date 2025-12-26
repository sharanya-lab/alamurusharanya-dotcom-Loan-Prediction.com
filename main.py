from data_preprocessing import load_data, preprocess_data
from model_training import train_model, evaluate_model
from prediction import load_model, load_scaler, predict

def main():
    # 1) Load & preprocess dataset
    df = load_data("loan_data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col="Loan_Status")

    # 2) Train model
    model = train_model(X_train, y_train)

    # 3) Evaluate
    evaluate_model(model, X_test, y_test)

    # 4) Predict sample
    loaded_model = load_model()
    loaded_scaler = load_scaler()
    example = X_test[0]  # use a test example as sample
    print("Sample prediction:", predict(loaded_model, loaded_scaler, example))

if __name__ == "__main__":
    main()

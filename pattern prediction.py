import numpy as np
import matplotlib.pyplot as plt
from lib.ltsm import LSTM
from lib.Random_forest import *
import xgboost as xgb  # Import XGBoost library
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA model

# Function to split data into training and testing sets
def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# Prepare training data
def prepare_data(data, window_size=5):
    normalized_data = data / 100
    ask = np.zeros((len(data) - window_size, window_size))
    answer = np.zeros((len(ask), 1))
    for i in range(len(ask)):
        ask[i] = normalized_data[i:i + window_size]
        answer[i] = normalized_data[i + window_size]
    return ask, answer

# Update function for pattern prediction
def update(index, results, coeffs, temp_data, predict_list, n=1, row_num=0):
    results[temp_data[index + 1]] += coeffs[row_num] * index / 100
    if n == 4:
        return
    if data[index - n] == predict_list[-n - 1]:
        update(index, results, coeffs, temp_data, predict_list, n=n + 1, row_num=row_num + 1)

# Train Random Forest
def train_random_forest(ask, answer, n_trees=100, max_depth=10, min_size=5, n_features=2, sample_size=0.8):
    forest = []
    for _ in range(n_trees):
        indices = np.random.choice(range(len(ask)), int(sample_size * len(ask)), replace=True)
        sample_train = ask[indices]
        sample_target = answer[indices]
        forest.append(build_tree(np.hstack((sample_train, sample_target)), max_depth, min_size, n_features))
    return forest

# Train XGBoost model
def train_xgboost(ask, answer, test_ask, test_answer):
    dtrain = xgb.DMatrix(ask, label=answer.ravel())
    dtest = xgb.DMatrix(test_ask, label=test_answer.ravel())  # Validation data

    params = {
        'objective': 'reg:squarederror',  # Regression task
        'max_depth': 5,                  # Optimal depth for balanced complexity
        'eta': 0.1,                     # Reduced learning rate for better generalization
        'subsample': 0.7,                # Row sampling for robustness
        'colsample_bytree': 0.7,         # Feature sampling for robustness
        'seed': 42,
        'lambda': 1.5,                   # L2 regularization for avoiding overfitting
        'alpha': 0.1                     # L1 regularization
    }
    num_round = 200  # Increased for better performance
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_round, evals=evals, early_stopping_rounds=20, verbose_eval=False)
    return model

# Train ARIMA model
def train_arima(train_data, order=(5, 1, 0)):
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    return fitted_model

# Evaluation metrics and predictions
def evaluate(data, test_data, train_data, model, forest, xgb_model, arima_model, coeffs):
    total_pattern = total_lstm = total_rf = total_xgb = total_arima = total = guessed_together = 0
    results = np.zeros(100)

    for i in range(len(test_data) - 5):
        add = 0
        
        temp_data = test_data[:i + 5]
        results.fill(0)
        predict_list = temp_data[-5:]
        indices = np.nonzero(np.array(train_data) == predict_list[-1])[0][0:-1]

        # Update results based on pattern matching
        for ind in indices:
            update(ind, results, coeffs, np.append(train_data, temp_data), predict_list)

        if list(results).index(max(results)) == data[i + 5]:
            total_pattern += 1
            add = 1

        # LSTM Prediction
        LSTM_predict = abs(model.forward_mult(predict_list / 100) * 100)
        results[int(LSTM_predict)] += .5
        if int(LSTM_predict) == data[i + 5]:
            total_lstm += 1
            add = 1

        # Random Forest Prediction
        rf_predict = bagging_predict(forest, predict_list / 100)
        rf_number = int(round(rf_predict * 100))
        results[rf_number] += 1
        if rf_number == data[i + 5]:
            total_rf += 1
            add = 1

        # XGBoost Prediction
        xgb_input = xgb.DMatrix(np.array(predict_list).reshape(1, -1) / 100)
        xgb_predict = xgb_model.predict(xgb_input)
        xgb_number = int(round(xgb_predict[0] * 100))
        results[xgb_number] += .5
        if xgb_number == data[i + 5]:
            total_xgb += 1
            add = 1

        # ARIMA Prediction
        arima_forecast = arima_model.forecast(steps=i+1)[-1]
        arima_number = int(arima_forecast)
        results[arima_number] += .5
        if arima_number == data[i + 5]:
            total_arima += 1
            add = 1

        if list(results).index(max(results)) == data[i + 5]:
            guessed_together += add
        total += add 
        
    return total_pattern, total_lstm, total_rf, total_xgb, total_arima, total, guessed_together

# Main execution
def main(data):
    # Hyperparameters
    coeffs = [1, 1.1, 1.5, 1.5, 1.5]
    n_trees = 5  # Increased number of trees
    max_depth = 7
    min_size = 1
    sample_size = 1
    n_features = 3
    it_count = 50000  # Reduced for computational efficiency
    lr = 0.0001

    # Split data
    train_data, test_data = split_data(data)

    # Prepare training data
    ask, answer = prepare_data(train_data)
    
    # Prepare validation data
    test_ask, test_answer = prepare_data(test_data)

    # Initialize LSTM model
    model = LSTM(lr, it_count)

    # Train Random Forest
    forest = train_random_forest(ask, answer, n_trees, max_depth, min_size, n_features, sample_size)

    # Train XGBoost with validation data
    xgb_model = train_xgboost(ask, answer, test_ask, test_answer)

    # Train ARIMA model
    arima_model = train_arima(train_data)

    # Evaluate on test data
    total_pattern, total_lstm, total_rf, total_xgb, total_arima, total, guessed_together = evaluate(
        data, test_data, train_data, model, forest, xgb_model, arima_model, coeffs
    )

    # Print results
    print(f"LSTM Accuracy: {total_lstm}, Percentage: {total_lstm / (len(test_data)-5) * 100:.2f}%")
    print(f"Pattern Matching Accuracy: {total_pattern}, Percentage: {total_pattern / (len(test_data)-5) * 100:.2f}%")
    print(f"Random Forest Accuracy: {total_rf}, Percentage: {total_rf / (len(test_data)-5) * 100:.2f}%")
    print(f"XGBoost Accuracy: {total_xgb}, Percentage: {total_xgb / (len(test_data)-5) * 100:.2f}%")
    print(f"ARIMA Accuracy: {total_arima}, Percentage: {total_arima / (len(test_data)-5) * 100:.2f}%")
    #print(f"Overall Accuracy: {total}, Percentage: {total / (len(test_data)-5) * 100:.2f}%")
    print(f"Overall combined Accuracy: {guessed_together}, Percentage: {guessed_together / (len(test_data)-5) * 100:.2f}%")

    # Visualize data distribution
    ammount = np.zeros(100)
    for value in data:
        ammount[value] += 1
    plt.bar(np.arange(0, 100), ammount)
    plt.show()

# Read data from a text file
with open("rand_numbers_data.txt", "r") as file:
    data = file.read()

# Convert the data into a NumPy array
data = np.array([int(x) for x in data.split(",")])

# Execute the main function
main(data)

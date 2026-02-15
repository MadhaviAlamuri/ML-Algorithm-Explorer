// Code Examples for ML Algorithms
const codeExamples = {
    "OLS (Ordinary Least Squares)": {
        python: `# Simple Linear Regression with scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data: house sizes and prices
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
new_house = np.array([[1800]])
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.2f}")

# Model parameters
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")`,
        
        explanation: "This example predicts house prices based on size. The model learns a linear relationship: Price = Slope × Size + Intercept"
    },
    
    "Lasso Regression": {
        python: `# Lasso Regression for Feature Selection
from sklearn.linear_model import Lasso
import numpy as np

# Data with many features (some irrelevant)
X = np.random.randn(100, 10)  # 10 features
y = 3*X[:, 0] + 5*X[:, 2] + np.random.randn(100)  # Only features 0 and 2 matter

# Lasso automatically selects important features
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Check which features were selected
print("Feature coefficients:")
for i, coef in enumerate(lasso.coef_):
    if abs(coef) > 0.01:
        print(f"Feature {i}: {coef:.3f} (Important)")
    else:
        print(f"Feature {i}: {coef:.3f} (Ignored)")`,
        
        explanation: "Lasso automatically identifies that only features 0 and 2 are important, setting other coefficients close to zero."
    },
    
    "Logistic Regression": {
        python: `# Binary Classification with Logistic Regression
from sklearn.linear_model import LogisticRegression
import numpy as np

# Email features: [word_count, has_urgent, has_money]
X = np.array([
    [50, 0, 0],   # Normal email
    [200, 1, 1],  # Likely spam
    [30, 0, 0],   # Normal email
    [150, 1, 1],  # Likely spam
])
y = np.array([0, 1, 0, 1])  # 0 = Not Spam, 1 = Spam

# Train the classifier
clf = LogisticRegression()
clf.fit(X, y)

# Predict new email
new_email = np.array([[100, 1, 0]])
prediction = clf.predict(new_email)
probability = clf.predict_proba(new_email)

print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
print(f"Confidence: {probability[0][prediction[0]]*100:.1f}%")`,
        
        explanation: "Logistic Regression calculates the probability that an email is spam based on its features."
    },
    
    "K-Means": {
        python: `# Customer Segmentation with K-Means
from sklearn.cluster import KMeans
import numpy as np

# Customer data: [age, annual_spending]
customers = np.array([
    [25, 30000], [28, 35000], [45, 80000],
    [50, 85000], [23, 28000], [48, 82000],
    [30, 40000], [52, 90000], [26, 32000]
])

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customers)

# Display results
for i, customer in enumerate(customers):
    print(f"Customer {i+1}: Age {customer[0]}, "
          f"Spending ${customer[1]:,} → Group {clusters[i]}")

# Cluster centers
print("\\nCluster Centers:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Group {i}: Avg Age {center[0]:.0f}, "
          f"Avg Spending ${center[1]:,.0f}")`,
        
        explanation: "K-Means groups customers into segments based on age and spending patterns. Great for targeted marketing!"
    },
    
    "Decision Tree Classification": {
        python: `# Decision Tree for Loan Approval
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Features: [income, credit_score, employment_years]
X = np.array([
    [50000, 650, 2],   # Approved
    [30000, 600, 1],   # Denied
    [80000, 750, 5],   # Approved
    [45000, 620, 3],   # Denied
    [90000, 800, 7],   # Approved
])
y = np.array([1, 0, 1, 0, 1])  # 1 = Approved, 0 = Denied

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

# Predict for new applicant
applicant = np.array([[60000, 700, 4]])
decision = tree.predict(applicant)
print(f"Loan Decision: {'Approved' if decision[0] == 1 else 'Denied'}")

# Feature importance
features = ['Income', 'Credit Score', 'Employment Years']
for name, importance in zip(features, tree.feature_importances_):
    print(f"{name}: {importance:.2%} importance")`,
        
        explanation: "Decision Trees make decisions by asking a series of yes/no questions about the features."
    },
    
    "Random Forest Classification": {
        python: `# Random Forest for Fraud Detection
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Transaction features: [amount, time_of_day, location_change]
X_train = np.array([
    [50, 14, 0],     # Normal
    [5000, 3, 1],    # Fraud
    [100, 10, 0],    # Normal
    [8000, 2, 1],    # Fraud
    [75, 18, 0],     # Normal
])
y_train = np.array([0, 1, 0, 1, 0])  # 0 = Normal, 1 = Fraud

# Train Random Forest (100 trees)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Check new transaction
new_transaction = np.array([[6000, 4, 1]])
prediction = rf.predict(new_transaction)
probability = rf.predict_proba(new_transaction)

print(f"Prediction: {'FRAUD ALERT' if prediction[0] == 1 else 'Normal'}")
print(f"Fraud Probability: {probability[0][1]*100:.1f}%")`,
        
        explanation: "Random Forest combines many decision trees to make more accurate predictions. Perfect for catching fraud!"
    },
    
    "Naive Bayes": {
        python: `# Naive Bayes for Sentiment Analysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Training data
reviews = [
    "This product is amazing and wonderful",
    "Terrible quality, very disappointed",
    "Love it! Best purchase ever",
    "Waste of money, do not buy",
    "Excellent product, highly recommend"
]
sentiments = [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X, sentiments)

# Predict sentiment of new review
new_review = ["Great product but expensive"]
X_new = vectorizer.transform(new_review)
prediction = nb.predict(X_new)
probability = nb.predict_proba(X_new)

print(f"Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {probability[0][prediction[0]]*100:.1f}%")`,
        
        explanation: "Naive Bayes is super fast for text classification. It calculates probabilities based on word frequencies."
    },
    
    "CNN (Convolutional Neural Network)": {
        python: `# Simple CNN for Image Classification
import tensorflow as tf
from tensorflow.keras import layers, models

# Build a simple CNN
model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Train on your image data
# model.fit(X_train, y_train, epochs=10)`,
        
        explanation: "CNNs automatically learn to detect edges, shapes, and patterns in images. Each layer learns increasingly complex features."
    },
    
    "LSTM (Long Short-Term Memory)": {
        python: `# LSTM for Text Generation
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Simple LSTM model
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(10000, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print("LSTM Model for Sequence Prediction")
print(f"Total parameters: {model.count_params():,}")

# Example: Predict next word in sequence
# Given: "The cat sat on the"
# Predict: "mat" (most likely next word)`,
        
        explanation: "LSTMs remember long-term patterns in sequences. Perfect for text, speech, and time series!"
    },
    
    "ARIMA": {
        python: `# ARIMA for Sales Forecasting
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

# Monthly sales data
sales = np.array([100, 120, 115, 130, 125, 140, 135, 150, 145, 160, 155, 170])
dates = pd.date_range('2023-01', periods=12, freq='M')
sales_series = pd.Series(sales, index=dates)

# Fit ARIMA model (p=1, d=1, q=1)
model = ARIMA(sales_series, order=(1, 1, 1))
fitted_model = model.fit()

# Forecast next 3 months
forecast = fitted_model.forecast(steps=3)

print("Historical Sales:", sales[-3:])
print("Forecasted Sales:", forecast.values.round(0))

# Model summary
print("\\nModel Summary:")
print(fitted_model.summary())`,
        
        explanation: "ARIMA captures trends and patterns in time series data. Great for forecasting sales, stock prices, or weather!"
    },
    
    "Collaborative Filtering": {
        python: `# Simple Collaborative Filtering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-Movie ratings matrix (0 = not rated)
ratings = np.array([
    [5, 3, 0, 1],  # User 1
    [4, 0, 0, 1],  # User 2
    [1, 1, 0, 5],  # User 3
    [0, 0, 5, 4],  # User 4
])

# Calculate user similarity
user_similarity = cosine_similarity(ratings)

# Recommend for User 1
user_id = 0
similar_users = user_similarity[user_id].argsort()[::-1][1:]  # Exclude self

print(f"User {user_id+1} similarity scores:")
for sim_user in similar_users:
    print(f"  User {sim_user+1}: {user_similarity[user_id][sim_user]:.3f}")

# Find movies User 1 hasn't rated but similar users liked
unwatched = np.where(ratings[user_id] == 0)[0]
print(f"\\nRecommended movies for User 1: {unwatched + 1}")`,
        
        explanation: "Collaborative Filtering finds users with similar tastes and recommends what they liked. This is how Netflix and Spotify work!"
    }
};

// Export for use in main script
window.codeExamples = codeExamples;

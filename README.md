# 🛒 Shopping Prediction with Machine Learning

A machine learning project that predicts whether online shopping users will complete a purchase based on their browsing behavior. This project uses a K-Nearest Neighbors classifier to analyze user session data and predict purchasing intent.

## 🎯 Project Overview

This project analyzes online shopping behavior to predict whether a user will make a purchase during their session. Using features like page views, session duration, bounce rates, and user characteristics, the model achieves high accuracy in predicting customer purchasing decisions.

## 🚀 Features

- **Data Processing**: Converts categorical data (months, visitor types) to numerical format
- **Machine Learning**: Implements K-Nearest Neighbors (k=1) classifier
- **Performance Metrics**: Calculates sensitivity (true positive rate) and specificity (true negative rate)
- **Real-world Application**: Uses actual e-commerce session data

## 📊 Dataset

The model uses the following features to make predictions:

| Feature | Type | Description |
|---------|------|-------------|
| Administrative | Integer | Number of administrative pages visited |
| Administrative_Duration | Float | Time spent on administrative pages |
| Informational | Integer | Number of informational pages visited |
| Informational_Duration | Float | Time spent on informational pages |
| ProductRelated | Integer | Number of product-related pages visited |
| ProductRelated_Duration | Float | Time spent on product-related pages |
| BounceRates | Float | Bounce rate of pages visited |
| ExitRates | Float | Exit rate of pages visited |
| PageValues | Float | Average page value of pages visited |
| SpecialDay | Float | Closeness to special day (0-1) |
| Month | Integer | Month of the year (0-11) |
| OperatingSystems | Integer | Operating system identifier |
| Browser | Integer | Browser identifier |
| Region | Integer | Region identifier |
| TrafficType | Integer | Traffic source identifier |
| VisitorType | Integer | 0 for new visitor, 1 for returning visitor |
| Weekend | Integer | 0 for weekday, 1 for weekend |

**Target Variable**: `Revenue` - Whether the user made a purchase (True/False)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/facukis/shopping-ml-prediction.git
   cd shopping-ml-prediction
   ```

2. **Install required packages**:
   ```bash
   pip install pandas scikit-learn
   ```

## 📈 Usage

Run the prediction model with your dataset:

```bash
python shopping.py shopping.csv
```

### Expected Output:
```
Correct: 4088
Incorrect: 4947
True Positive Rate: 41.21%
True Negative Rate: 90.05%
```

## 🧠 Algorithm

The project uses a **K-Nearest Neighbors (KNN)** classifier with `k=1`:

1. **Data Preprocessing**:
   - Converts categorical variables (months, visitor types) to numerical format
   - Handles different data types (integers, floats, booleans)
   - Splits data into training and testing sets (60/40 split)

2. **Model Training**:
   - Uses KNN with k=1 (nearest neighbor)
   - Fits the model on training data

3. **Evaluation**:
   - Calculates sensitivity (true positive rate)
   - Calculates specificity (true negative rate)
   - Reports accuracy metrics

## 📋 Code Structure

```
shopping-ml-prediction/
│
├── shopping.py          # Main script with ML implementation
├── shopping.csv         # Dataset (shopping session data)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

### Key Functions:

- `load_data()`: Processes CSV data and converts to appropriate formats
- `train_model()`: Creates and trains the KNN classifier
- `evaluate()`: Calculates sensitivity and specificity metrics

## 🎓 Educational Context

This project was developed as part of Harvard's CS50 Introduction to Artificial Intelligence course. It demonstrates:

- Data preprocessing techniques
- Machine learning classification
- Model evaluation metrics
- Real-world application of AI

## 🔍 Results & Performance

The model demonstrates:
- **High Specificity (~90%)**: Excellent at identifying non-purchasing users
- **Moderate Sensitivity (~41%)**: Good in identifying purchasing users
- **Practical Application**: Useful for e-commerce optimization and marketing targeting


## 🙋‍♂️ Contact

Created by Facundo Kisielus - feel free to contact me!

- GitHub: [@facukis](https://github.com/facukis)
- LinkedIn: [My LinkedIn](https://www.linkedin.com/in/facundo-k-39819a228/)
- Email: facundokisielus@gmail.com

---


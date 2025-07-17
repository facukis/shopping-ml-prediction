
import sys
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv(filename)
    # we check if user bought a product or not and put it into a list of 1 or 0
    labels = df['Revenue']
    labels_num = labels.astype(int)
    labels_list = labels_num.tolist()
    
    evidence = df.drop(columns=['Revenue'])
    # we transform the evidence data in convenient data types
    evidence['Administrative'] = evidence['Administrative'].astype(int)
    evidence['Informational'] = evidence['Informational'].astype(int)
    evidence['ProductRelated'] = evidence['ProductRelated'].astype(int)
    evidence['OperatingSystems'] = evidence['OperatingSystems'].astype(int)
    evidence['Browser'] = evidence['Browser'].astype(int)
    evidence['Region'] = evidence['Region'].astype(int)
    evidence['TrafficType'] = evidence['TrafficType'].astype(int)
    evidence['BounceRates'] = evidence['BounceRates'].astype(float)
    evidence['Weekend'] = evidence['Weekend'].astype(int)
    evidence['Administrative_Duration'] = evidence['Administrative_Duration'].astype(float)
    evidence['Informational_Duration'] = evidence['Informational_Duration'].astype(float)
    evidence['ProductRelated_Duration'] = evidence['ProductRelated_Duration'].astype(float)
    evidence['ExitRates'] = evidence['ExitRates'].astype(float)
    evidence['PageValues'] = evidence['PageValues'].astype(float)
    evidence['SpecialDay'] = evidence['SpecialDay'].astype(float)
    evidence['Month'] = evidence['Month'].replace({'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11})
    evidence['VisitorType'] = evidence['VisitorType'].apply(lambda x: 1 if x == 'Returning_Visitor' else 0)
 
    evidence_list = evidence.values.tolist()

    return (evidence_list, labels_list)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    X_training = evidence
    y_training = labels
    model.fit(X_training, y_training)
    
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    correct_sensitivity = 0
    incorrect_sensitivity = 0
    total_sensitivity = 0

    correct_specificity = 0
    incorrect_specificity = 0
    total_specificity = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_sensitivity += 1
            if actual == predicted:
                correct_sensitivity += 1
            else:
                incorrect_sensitivity += 1
        
        elif actual == 0:
            total_specificity += 1
            if actual == predicted:
                correct_specificity += 1
            else:
                incorrect_specificity += 1

    sensitivity = correct_sensitivity/total_sensitivity
    specificity = correct_specificity/total_specificity

    return sensitivity, specificity

if __name__ == "__main__":
    main()

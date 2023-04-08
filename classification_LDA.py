from utils import *
import time
def test_train_set(data):
    # Split data into features and labels
    features = data.drop(columns=['label'])
    # normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    labels = data['label']

    feature_train,feature_test,label_train,label_test = train_test_split(features,labels,test_size=0.10)
    #print(label_test)
    return feature_train,feature_test,label_train,label_test

def train_test_valid_LDA(data):

    feature_train,feature_test,label_train,label_test = test_train_set(data)

    # create the model
    model = LinearDiscriminantAnalysis()

    # fit the model to the training data
    model.fit(feature_train, label_train)

    # make predictions on the test data
    prediction = model.predict(feature_test)
    # Compute the accuracy of the classifier on the test data
    accuracy = accuracy_score(label_test, prediction)
    print(f'Accuracy LDA: {accuracy:.2f}')

    # Calculate the accuracy for each class
    #classes = list(set(label_test))  # Get the unique class labels
    #for c in classes:
        # Select the examples for this class
    #    y_true_class = [y == c for y in label_test]
    #    y_pred_class = [y == c for y in prediction]

        # Calculate the accuracy for this class
    #    accuracy_class = accuracy_score(y_true_class, y_pred_class)

    #    print("Accuracy for class {}: {}".format(c, accuracy_class))

    # Generate the confusion matrix
    cm = confusion_matrix(label_test, prediction, labels=model.classes_)

    # Convert the counts in the confusion matrix to percentages
    cm = (100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    cm = cm.astype('int')

    #Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = model.classes_)
    #Plot confusion matrix
    disp.plot()
    plt.show()

def train_test_valid_MLP(data,hlayers,neurons,maxIter,activation,dataset):
    feature_train,feature_test,label_train,label_test = test_train_set(data)
    # create an instance of LabelEncoder
    le = LabelEncoder()

    # fit the encoder to the label data and transform it
    y_train_encoded = le.fit_transform(label_train)
    y_test_encoded = le.transform(label_test)
    # Create an MLP classifier with 2 hidden layers of 5 neurons each
    if hlayers == 2:
        model = MLPClassifier(hidden_layer_sizes=(neurons,neurons), max_iter=maxIter,activation=activation, random_state=1)
    if hlayers == 4:
        model = MLPClassifier(hidden_layer_sizes=(neurons, neurons), max_iter=maxIter, activation=activation, random_state=1)
    if hlayers == 6:
        model = MLPClassifier(hidden_layer_sizes=(neurons, neurons), max_iter=maxIter, activation=activation, random_state=1)
    # Train the classifier on the training data
    model.fit(feature_train, y_train_encoded)

    #Current Timstamp to seperate model names
    timestamp = int(time.time() * 1000)
    #set name for model and save it
    filename = f"MLP_Models/MLP_model_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}_{timestamp}.joblib"
    joblib.dump(model, filename)
    # make predictions on the test data
    prediction = model.predict(feature_test)

    # Evaluate the accuracy of the classifier on the test data
    accuracy = accuracy_score(y_test_encoded, prediction)

    print(f'Accuracy MLP_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}: {accuracy:.2f}')

    # transform labels back to strings
    inverse_pred=le.inverse_transform(prediction)
    inverse_test_labels=le.inverse_transform(y_test_encoded)
    # Generate the confusion matrix
    cm = confusion_matrix(inverse_test_labels, inverse_pred, labels=le.inverse_transform(model.classes_))

    # Convert the counts in the confusion matrix to percentages
    cm = (100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    cm = cm.astype('int')

    print(f'Confusion Matrix MLP_{hlayers}_{neurons}_{maxIter}_{activation}_dataset{dataset}:\n {cm}')
    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.inverse_transform(model.classes_))
    # Plot confusion matrix
    disp.plot()
    plt.show()
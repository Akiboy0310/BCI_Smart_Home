from utils import *
def test_train_set(data):
    # Split data into features and labels
    features = data.drop(columns=['label'])
    # normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    print(f"Hallo Warum siehst du mich nicht: {scaled_data}")
    labels = data['label']
    #print(f"Features are : {features}")
    #print(f'This are Labels: {labels}')
    feature_train,feature_test,label_train,label_test = train_test_split(features,labels,test_size=0.20)

    return feature_train,feature_test,label_train,label_test

def train_test_valid_LDA(data):

    feature_train,feature_test,label_train,label_test = test_train_set(data)

    # create the model
    model = LinearDiscriminantAnalysis()


    print(f"this are labels:{label_train}")
    # fit the model to the training data
    model.fit(feature_train, label_train)

    # make predictions on the test data
    prediction = model.predict(feature_test)
    # Compute the accuracy of the classifier on the test data
    accuracy = accuracy_score(label_test, prediction)
    print(f'Accuracy: {accuracy:.2f}')

    # Calculate the accuracy for each class
    classes = list(set(label_test))  # Get the unique class labels
    for c in classes:
        # Select the examples for this class
        y_true_class = [y == c for y in label_test]
        y_pred_class = [y == c for y in prediction]

        # Calculate the accuracy for this class
        accuracy_class = accuracy_score(y_true_class, y_pred_class)

        print("Accuracy for class {}: {}".format(c, accuracy_class))

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


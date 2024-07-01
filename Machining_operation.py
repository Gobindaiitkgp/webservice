import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn import svm
from tensorflow import keras
from tensorflow.keras import layers


# Load dataset
data = pd.read_csv('MachiningOperation (1).csv')
# Separate features and target
x = data.drop(columns=['MachiningOperation'], axis=1)
y = data['MachiningOperation']

# Label encoding for target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Initialize models
log_reg = LogisticRegression(C=100, random_state=1, solver='lbfgs', multi_class='ovr')
svc_model = SVC(kernel='linear', probability=True, random_state=0)
RandomForest_model = RandomForestClassifier(criterion="gini", max_depth=10, min_samples_split=2, random_state=0, n_estimators=100, max_features="sqrt")
DecisionTree_model = DecisionTreeClassifier(criterion='gini', min_samples_split=2)
gaussianNB_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Train a neural network model
num_classes = len(np.unique(y))  # Ensure the number of classes is correctly calculated
input_shape = (x_train.shape[1],)
# Train a neural network model

# Convert labels to one-hot encoding for neural network
y_train_nn = keras.utils.to_categorical(y_train, num_classes)
y_test_nn = keras.utils.to_categorical(y_test, num_classes)

# Define the neural network architecture
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=input_shape),
    layers.Dense(20, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train the neural network model
epochs = 100
batch_size = 64
history = model.fit(x_train, y_train_nn, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_nn))

# Train other models
log_reg.fit(x_train, y_train)
svc_model.fit(x_train, y_train)
RandomForest_model.fit(x_train, y_train)
DecisionTree_model.fit(x_train, y_train)
gaussianNB_model.fit(x_train, y_train)
knn_model.fit(x_train, y_train)

# Save models
pickle.dump(log_reg, open('log_model.pkl', 'wb'))
pickle.dump(svc_model, open('svc_model.pkl', 'wb'))
pickle.dump(RandomForest_model, open('RandomForest_model.pkl', 'wb'))
pickle.dump(DecisionTree_model, open('DecisionTree_model.pkl', 'wb'))
pickle.dump(gaussianNB_model, open('gaussianNB_model.pkl', 'wb'))
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
model.save('neural_network_model.h5')

# Load models
log_model = pickle.load(open('log_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
RandomForest_model = pickle.load(open('RandomForest_model.pkl', 'rb'))
DecisionTree_model = pickle.load(open('DecisionTree_model.pkl', 'rb'))
gaussianNB_model = pickle.load(open('gaussianNB_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
nn_model = keras.models.load_model('neural_network_model.h5')

def classify(prediction):
    operations = [
        'Drilling', 'Drilling-CounterBoring', 'Drilling-CounterBoring-FinishBroaching',
        'Drilling-CounterBoring_RoughReaming_SemifinishReaming', 'Drilling-CounterBoring_RoughReaming_SemifinishReaming',
        'Drilling-RoughBoring-SemifinishBoring', 'Drilling-RoughBoring-SemifinishBoring-DiamondBoring',
        'Drilling-RoughBoring-SemifinishBoring-FinishBoring', 'Drilling-RoughBoring-SemifinishBoring-Grinding-Honing',
        'Drilling-RoughBoring-SemifinishBoring-Grinding-Lapping', 'Drilling-RoughBoring-SemifinishBoring-RoughGrinding-FinishGrinding',
        'Drilling-RoughBoring-SemifinishBoring-RoughGrinding-SemifinishGrinding', 'RoughMilling', 'RoughMilling-SemifinishMilling',
        'RoughMilling-SemifinishMilling-FinishMilling', 'RoughMilling-SemifinishMilling-Grinding-Lapping', 'RoughMilling-SemifinishMilling-RoughGrinding',
        'RoughMilling-SemifinishMilling-Grinding-Superfinishing'
    ]
    return operations[prediction]

def app():
    html_temp = """
    <div style="background-color:teal; padding:10px">
    <h2 style="color:white; text-align:center;">Machining Operations</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Logistic Regression', 'SVC', 'RandomForest Classifier', 'DecisionTree Classifier', 'GaussianNB', 'Kneighbors Classifier', 'Neural Network']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    
    ft = st.selectbox('Features:', list(range(1, 11)))
    
    D = st.slider('Select Diameter(mm)', 0.0, 250.0)
    Dp = st.slider('Select Depth(mm)', 0.0, 250.0)
    L = st.slider('Select Length(mm)', 0.0, 250.0)
    W = st.slider('Select Width(mm)', 0.0, 250.0)
    R = st.slider('Select Radius(mm)', 0.0, 30.0)
    A = st.slider('Select Angle(0)', 0.0, 90.0)
    Dis = st.slider('Select Distance(mm)', 0.0, 18.0)
    tl = st.slider('Select Tolerance(µm)', 0.0, 720.0)
    sf = st.slider('Select Surface Finish(µm)', 0.0, 80.0)
    inputs = [[ft, D, Dp, L, W, R, A, Dis, tl, sf]]
    
    if st.button('Classify'):
        if option == 'Logistic Regression':
            st.success(classify(log_model.predict(inputs)[0]))
        elif option == 'RandomForest Classifier':
            st.success(classify(RandomForest_model.predict(inputs)[0]))
        elif option == 'DecisionTree Classifier':
            st.success(classify(DecisionTree_model.predict(inputs)[0]))
        elif option == 'GaussianNB':
            st.success(classify(gaussianNB_model.predict(inputs)[0]))
        elif option == 'Kneighbors Classifier':
            st.success(classify(knn_model.predict(inputs)[0]))
        elif option == 'Neural Network':
            prediction = nn_model.predict(np.array(inputs))
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            st.success(predicted_class)
        else:
            st.success(classify(svc_model.predict(inputs)[0]))

if __name__ == '__main__':
    app()

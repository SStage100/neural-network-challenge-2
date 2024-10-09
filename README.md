# neural-network-challenge-2
Module  Challenge
# neural-network-challenge-2
Module  Challenge
Neural Network Challenge - Employee Attrition Prediction

Welcome! This project focuses on using a neural network model to predict whether employees are likely to leave a company (attrition) and which department they are best suited for. I completed this project as part of a neural network challenge, and I’m excited to walk you through how I tackled it.

Overview
In this project, I built a branched neural network that helps HR predict two things:

1. Employee Attrition: Whether an employee is likely to leave the company.
2. Department Prediction: Which department the employee would best fit into.

By training a neural network model on employee data, we can make predictions that might help HR teams make better decisions!

Project Structure
Here’s a breakdown of the project:

1. Data Preprocessing: The employee data was cleaned and transformed so that it could be used to train the neural network.
2. Model Building: A neural network with two output branches was built using TensorFlow/Keras.
3. Training and Evaluation: The model was trained, and then evaluated on test data to check its performance.
4. Results: The accuracy for predicting both attrition and department was evaluated.

Files
1. attrition.ipynb: This is the main Jupyter notebook where I performed all the data preprocessing, built the neural network, and trained/evaluated the model.

Data Preprocessing
Before diving into the model, I had to prepare the data. The data came in a CSV file with employee details, such as age, job satisfaction, number of companies worked for, and more. Here's what I did:

1. Selected Features: I chose 10 columns to use as input features for the model. These included things like age, education, and job satisfaction.

2. Encoding: Some of the data, like whether an employee travels for work or if they work overtime, had to be converted into numbers (this is called "encoding"). I used one-hot encoding for this.

3. Scaling: To make sure all the input data was on a similar scale, I used something called StandardScaler. This helps the model train better.

Model Building
For the model, I used TensorFlow and Keras to build a branched neural network. Here’s a quick rundown of the structure:

1. Input Layer: The input features were fed into the network.
2. Shared Hidden Layers: These layers are used by both branches of the model to learn common patterns in the data.
3. Branch for Department: One part of the model focuses on predicting which department the employee belongs to.
4. Branch for Attrition: The other part of the model predicts whether an employee will leave or stay.

Activation Functions
For both branches, I used the softmax activation function in the output layers. This function is useful for multi-class classification, like choosing the right department or whether someone will leave the company.

Training the Model
The model was trained using employee data. I split the data into training and testing sets so that the model could learn from one part of the data and then be tested on data it hasn’t seen before. The model was trained for 50 epochs using a batch size of 32.

Evaluation
Once the model was trained, I evaluated it using the test data. The evaluation gave me the accuracy for both the department prediction and the attrition prediction.

Results
1. Department Prediction Accuracy: After training, the model could predict which department an employee belongs to with reasonable accuracy.

2. Attrition Prediction Accuracy: The model also gave a pretty good indication of whether employees might leave the company.

Improvements
While the model worked, there’s always room for improvement. Some things I could try next:

1. Tuning the model: Adjusting the layers, neurons, or learning rate could give better results.

2. Handling Imbalanced Data: If there’s a lot more data for one group (like employees who stay), using techniques to handle this imbalance could help the model learn better.

3. Regularization: This can help prevent the model from overfitting (memorizing the training data rather than learning to generalize).

How to Run the Code
To run this project:

Clone this repository.

Open attrition.ipynb in Google Colab (or locally if you prefer).
Install the necessary dependencies (TensorFlow, Pandas, Scikit-learn).
Run the notebook, starting with the data preprocessing all the way through to model training and evaluation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree._reingold_tilford import move_subtree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


filePath = 'C:\\Users\\islei\\Downloads\\Diabetes.csv'
data = pd.read_csv(filePath)

def replace_outliers_and_zeros_with_median(column):#this function to reblace value that unader or more outlier
    quantile1 = column[column > 0].quantile(0.25)
    quantile3 = column[column > 0].quantile(0.75)
    iqr = quantile3 - quantile1
    lower_limit = quantile1 - 1.5 * iqr
    upper_limit = quantile3 + 1.5 * iqr
    column_copy = column.copy()
    column_copy[(column_copy < lower_limit) | (column_copy > upper_limit) | (column_copy == 0)] = column.median()
    return column_copy


columns_to_replace = ["PGL", "DIA", "TSF", "INS", "BMI", "DPF", "AGE"]#i will just replace data from PGL to AGE

for column_name in columns_to_replace:
    data[column_name] = replace_outliers_and_zeros_with_median(data[column_name])


#print(data)---to print data for chicking

print("Choice one of thesee options :\n1.PART1 \n2.PART2 \n3.PART3\n4.EXIT ")
part=input()
part=int(part)
while(part!=4):

    if(part==1):
        print("PART1:")

        print("Enter one of these options : ")
        print("1. Print the summary statistics of all attributes in the dataset")
        print("2. Show the distribution of the class label 'Diabetic' and highlight it")
        print("3.draw the histogram for group age ")
        print("4.Density Plot for Age")
        print("5.density Plot for BMI")
        print("6.Correlation Between Features")
        print("7.Split the data set into traning(80%) and test(20%)")
        print("8.exit")

        number = input()
        number = int(number)
        while(number!=8):

            if number == 1:
                summaryStatistics = data.describe()
                print(summaryStatistics)

            elif number == 2:
                classCounts = data['Diabetic'].value_counts()
                plt.figure(figsize=(6, 4))
                sns.countplot(x="Diabetic", data=data, palette='viridis')
                plt.title("Distribution of Diabetic")
                plt.xlabel("Diabetice")
                plt.ylabel("Count")
                # display the counts above the bar
                for i, count in enumerate(classCounts):
                    plt.text(i, count + 0.1, str(count), ha='center')

                plt.show()
            elif number == 3:
                plt.figure(figsize=(10, 6))
                sns.histplot(data[data['Diabetic'] == 1]["AGE"], bins=20, kde=False, color='blue',
                             label='Diabetic')  # حساب عدد اعمار الاشخاص المصابين لمرض السكري
                sns.histplot(data[data['Diabetic'] == 0]["AGE"], bins=20, kde=False, color='black', label='Non-Diabetic')
                plt.title('Diabetics and Non-Diabetics in Each Age Group')
                plt.xlabel('Age')
                plt.ylabel('Count')
                plt.legend()
                plt.show()
            elif number == 4:
                plt.figure(figsize=(18, 6))
                sns.kdeplot(data['AGE'], fill=True, color='blue')
                plt.title("Density Plot for Age")
                plt.xlabel("Age")
                plt.ylabel("Density")
                plt.show()

            elif number == 5:

                plt.figure(figsize=(11, 6))
                sns.kdeplot(data["BMI"], fill=True, color="black")
                plt.title("Density plot for BMI")
                plt.xlabel("BMI")
                plt.ylabel("Density")
                plt.show()
            elif number == 6:
                TheCorrelation = data.corr()
                # create correlation using the map
                plt.figure(figsize=(13, 7))
                sns.heatmap(TheCorrelation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title("Correlation Between all Features")
                plt.show()

            elif number == 7:


                X = data.iloc[:, :-1]  # Features(input data)
                y = data.iloc[:, -1]  # output value where is the last column in the data


                # Set a seed for reproducibility
                np.random.seed(42)

                # Generate random indices for training and test sets
                indices = np.random.permutation(len(data))
                split = int(0.8 * len(data))

                # Split the data
                X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
                y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]
                print(X_test)
                print(X_train)

                """
                from above input u can relesed there is 80 of data input is traninig and 20 of daata is test where you can see the size of row 
                and same in y traning and test 
                """
            print("Enter one of these options : ")
            print("1. Print the summary statistics of all attributes in the dataset")
            print("2. Show the distribution of the class label 'Diabetic' and highlight it")
            print("3.draw the histogram for group age ")
            print("4.Density Plot for Age")
            print("5.density Plot for BMI")
            print("6.Correlation Between Features")
            print("7.Split the data set into traning(80%) and test(20%)")
            print("8.exit")

            number = input()
            number = int(number)

    elif part==2:
        print("PART2:")
        number=input("choice:\n1.Apply linear regression to learn the attribute Age using allindependent attributes\n2.Apply linear regression using the most important feature\n3.Apply linear regression using the set of most important features\n4.EXIT")
        number=int(number)
        while(number!=4):

            if number == 1:
                X = data.drop('AGE', axis=1)  # Exclude the target variable 'AGE'
                y = data['AGE']

                # Set a seed for reproducibility
                np.random.seed(42)
                # Generate random indices for training and test sets
                indices = np.random.permutation(len(data))
                split = int(0.8 * len(data))

                # Split the data into training and testing sets (80% training, 20% testing)
                X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
                y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

                # Create and train the linear regression model
                leniar_model = LinearRegression()
                leniar_model.fit(X_train, y_train)

                # Make predictions on the test set
                Y_Predict = leniar_model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, Y_Predict)
                print(f'Mean Squared Error: ', mse)

                # Visualize the actual vs predicted values
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_test, y=Y_Predict)
                plt.xlabel('Actual Age')
                plt.ylabel('Predicted Age')
                plt.title('Actual vs Predicted Age')
                plt.show()


            elif number == 2:
                correlation_matrix = data.corr()
                age_correlations = correlation_matrix['AGE'].abs().sort_values(ascending=False)#using this statment i can see what is best correlation with age
                # print(age_correlations)
                most_important_feature = age_correlations.index[1]
                # print(most_important_feature)#in this case i will do correlaction with NPG  beacse its have high percentage with target value (age)

                print("The most important feature for predicting age is " + most_important_feature)
                print(most_important_feature)
                X = data[[most_important_feature]]
                y = data['AGE']

                # Set a seed for reproducibility
                np.random.seed(42)
                # Generate random indices for training and test sets
                indices = np.random.permutation(len(data))
                split = int(0.8 * len(data))

                # Split the data into training and testing sets (80% training, 20% testing)
                X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
                y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

                # Create and train the linear regression model
                leniar_model = LinearRegression()
                leniar_model.fit(X_train, y_train)

                # Make predictions on the test set
                Y_Predict = leniar_model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(y_test, Y_Predict)
                print(f'Mean Squared Error: ', mse)

                # Visualize the actual vs predicted values
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_test, y=Y_Predict)
                plt.xlabel('Actual Age')
                plt.ylabel('Predicted Age')
                plt.title('Actual vs Predicted Age')
                plt.show()

            elif number == 3:
                correlation_matrix = data.corr()
                top_3_features = correlation_matrix['AGE'].abs().sort_values(ascending=False).index[1:4]
                print(data[top_3_features].describe())
                print(f"the top 3 feature where its correlation matrix with age is {top_3_features}")  # same befor

                X = data[top_3_features]
                print(X)
                y = data['AGE']

                # Set a seed for reproducibility
                np.random.seed(42)
                # Generate random indices for training and test sets
                indices = np.random.permutation(len(data))
                split = int(0.8 * len(data))

                # Split the data into training and testing sets (80% training, 20% testing)
                X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
                y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

                # Create and train the linear regression model
                leniar_model = LinearRegression()
                leniar_model.fit(X_train, y_train)
                Y_Predict = leniar_model.predict(X_test)
                # Evaluate the model
                mse = mean_squared_error(y_test, Y_Predict)
                print(f'Mean Squared Error: ', mse)

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_test, y=Y_Predict)
                plt.xlabel('Actual Age')
                plt.ylabel('Predicted Age')
                plt.title('Actual vs Predicted Age')
                plt.show()

            number = input(
                "choice: 1.Apply linear regression to learn the attribute Age using allindependent attributes\n2.Apply linear regression using the most important feature\n3.Apply linear regression using the set of most important features\n4.EXIT")
            number = int(number)

    else:
        print("PART3")
        number=input("Choice:\n1.K-nearest-neighber using test set\n2.K-Neareast-neighber using at last 4 models\n3.EXIT")
        number=int(number)

        while(number!=3):

            if number == 1:
                X = data.drop('Diabetic', axis=1)
                y = data['Diabetic']
                # Set a seed for reproducibility
                np.random.seed(42)
                # Generate random indices for training and test sets
                indices = np.random.permutation(len(data))
                split = int(0.8 * len(data))

                # Split the data into training and testing sets (80% training, 20% testing)
                X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
                y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]
                # Create and train the k-NN classifier
                knn_model = KNeighborsClassifier(n_neighbors=9)  # You can adjust the value of k as needed
                knn_model.fit(X_train, y_train)
                # Make predictions on the test set
                y_pred_knn = knn_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_knn)
                accuracy *= 100
                accuracy = int(accuracy)
                auc = roc_auc_score(y_test, y_pred_knn)
                conf_matrix = confusion_matrix(y_test, y_pred_knn)
                print(f" Accuuracy : {accuracy} auc : {auc} and conf Matrix :{conf_matrix}  ")


            elif number == 2:
                X = data.drop('Diabetic', axis=1)
                y = data['Diabetic']
                np.random.seed(42)
                # Generate random indices for training and test sets
                indices = np.random.permutation(len(data))
                split = int(0.8 * len(data))

                # Split the data into training and testing sets (80% training, 20% testing)
                X_train, X_test = X.iloc[indices[:split]], X.iloc[indices[split:]]
                y_train, y_test = y.iloc[indices[:split]], y.iloc[indices[split:]]

                value_of_key = [ 1, 3, 5, 7]
                models_of_k = {}

                for i in value_of_key:
                    k_model = KNeighborsClassifier(n_neighbors=i)
                    k_model.fit(X_train, y_train)
                    models_of_k[i] = k_model

                # Evaluate each model
                for k, model in models_of_k.items():
                    # Make predictions on the test set
                    Y_Predict = model.predict(X_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, Y_Predict)
                    auc = roc_auc_score(y_test, Y_Predict)
                    conf_matrix = confusion_matrix(y_test, Y_Predict)
                    print(f"K : {k} Accuuracy : {accuracy} auc : {auc} and conf Matrix :{conf_matrix}  ")
            number = input(
                "Choice:\n1.K-nearest-neighber using test set\n2.K-Neareast-neighber using at last 4 models\n3.EXIT")
            number = int(number)

    print("Choice one of thesee options :\n1.PART1 \n2.PART2 \n3.PART3\n4.EXIT ")
    part = input()
    part = int(part)












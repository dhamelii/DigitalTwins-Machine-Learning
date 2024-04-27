import os
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import sqlalchemy

# Clean CSV files from Tensile Tester so they are all uniform and contain only required data
def tensile_remove_quotes_from_column_headers(csv_folder):
    for file in os.listdir(csv_folder):
        file_path = os.path.join(csv_folder, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            first_line = lines[0].strip()  # First line contains column headers
            new_first_line = ""
            in_quotes = False
            for char in first_line:
                if char == '"':
                    in_quotes = not in_quotes
                elif in_quotes and char == ',':
                    # Replace comma inside quotes with a placeholder
                    new_first_line += '<COMMA>'
                else:
                    new_first_line += char
            
            # Replace quotes from each column header
            new_first_line = new_first_line.replace('"', '')

            # Restore commas inside quotes
            new_first_line = new_first_line.replace('<COMMA>', ',')
            
            # Rewrite the file with modified first line
            with open(file_path, 'w') as f:
                f.write(new_first_line + '\n')
                # Write the remaining lines of the file
                for line in lines[1:]:
                    f.write(line)

# Clean CSV files from CyberGage360 so they are all uniform and contain only required data
def measurement_file_cleanup(csv_folder):

    for file_name in os.listdir(csv_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csv_folder, file_name)
            
            with open(file_path, 'r', newline='') as file:
                csv_reader = csv.reader(file)
                data = list(csv_reader)
            
            #Check if the first row is '##Cross Section1' and delete rows
            if data[0][0].strip('ï»¿') == '##Cross Section1':
                # Remove the first row and rows 48 and 49 (indices 47 and 48)
                data.pop(0)
                data.pop(48)
                data.pop(48)

            #Change I to i for measurement names, rearrange measurement name
            for row in data:
                row[0] = row[0].replace('I', 'i')
                if row[0].startswith('90_'):
                    row[0] = row[0][3:] + '_90'
                
                # Write the modified data back to the CSV file
            with open(file_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(data)
                    
            print(f"Processed {file_name} and saved back to the same file.")
        else:
            print(f"Skipping {file_name} as it does not require preprocessing.")

# Pull data from Tensile Testing and Dimensional Data from CyberGage360
def load_data(tensile_folder, inspection_folder):
    tensile_data = []
    inspection_data = []
    
    # Load tensile testing data
    for file in os.listdir(tensile_folder):
        bolt, test = extract_info_from_filename(file)
        df = pd.read_csv(os.path.join(tensile_folder, file))
        df['Bolt'] = bolt
        df['Test'] = test
        tensile_data.append(df)
    
    # Load dimensional inspection data
    for file in os.listdir(inspection_folder):
        bolt, test = extract_info_from_filename(file)
        df = pd.read_csv(os.path.join(inspection_folder, file))
        df['Bolt'] = bolt
        df['Test'] = test
        inspection_data.append(df)
    
    tensile_data = pd.concat(tensile_data)
    inspection_data = pd.concat(inspection_data)
    
    return tensile_data, inspection_data

# Pull bolt number and test number from file name
def extract_info_from_filename(filename):

    parts = filename.split("_")

    if len(parts) < 5:
        bolt = ''.join(filter(str.isdigit, parts[-2]))  # Extract bolt number from filename
        test = ''.join(filter(str.isdigit, parts[-1].split(".")[0]))  # Extract test number from filename
    
    else:
        bolt = parts[-3]  # Extract bolt number from filename
        test = parts[-1].split(".")[0]  # Extract test number from filename

    return bolt, test

# Merge Tensile and Dimensional Dataframes. Create Fracture column and load data.
# Handle Missing Values
def preprocess_data(tensile_data, inspection_data):

    # Remove columns critical to model from dimensional inspection.
    inspection_columns = ['Name', 'Meas. Value', 'Bolt', 'Test']
    inspection_data = inspection_data[inspection_columns]
    # print(inspection_data)

    # Choose specific measurements to focus on as entire bolt was not tested
    inspection_names = ['Angle_Left_1', 'Angle_Left_2', 'Angle_Left_3', 'Angle_Left_4', 'Angle_Right_1', 'Angle_Right_2',
                          'Angle_Right_3', 'Major_OD_1', 'Major_OD_2', 'Major_OD_3', 'Major_OD_4', 'Minor_OD_1', 'Minor_OD_2', 'Minor_OD_3',
                            'Minor_OD_4', 'Overall_Length', 'Pitch_Left_1', 'Pitch_Left_2', 'Pitch_Left_3', 'Pitch_Right_1', 'Pitch_Right_2',
                              'Pitch_Right_3', 'Angle_Left_1_90', 'Angle_Left_2_90', 'Angle_Left_3_90', 'Angle_Right_1_90', 'Angle_Right_2_90',
                                'Angle_Right_3_90', 'Major_OD_1_90', 'Major_OD_2_90', 'Major_OD_3_90', 'Minor_OD_1_90', 'Minor_OD_2_90', 'Minor_OD_3_90',
                                  'Minor_OD_4_90', 'Overall_Length_90', 'Pitch_Left_1_90', 'Pitch_Left_2_90', 'Pitch_Left_3_90', 'Pitch_Right_1_90', 'Pitch_Right_2_90',
                                    'Pitch_Right_3_90']
    
    # Filter inspection data to only show critical features
    inspection_data = inspection_data[inspection_data['Name'].isin(inspection_names)]

    # Remove columns critical to model from tensile testing.
    tensile_columns = [' Load (lb)', ' Position (in)', 'Bolt', 'Test']
    tensile_data = tensile_data[tensile_columns]

    # Calculate Max_Load, Max_Position, Max_Strain, Max_Stress, and Avg_Stress
    tensile_data_grouped = tensile_data.groupby(['Bolt', 'Test']).agg(
        Max_Load=(' Load (lb)', 'max'),
        Max_Position=(' Position (in)', 'max'),
    )

    # Calculate Max_Strain
    tensile_data_grouped['Max_Strain'] = (2.09 + tensile_data_grouped['Max_Position']) / 2.09

    # Calculate Max_Stress
    tensile_data_grouped['Max_Stress'] = tensile_data_grouped['Max_Load'] / 0.44

    # Calculate Avg_Stress
    tensile_data_grouped['Avg_Stress'] = tensile_data.groupby(['Bolt', 'Test'])[' Load (lb)'].mean() / 0.44

    # Create Fracture column
    fracture_conditions = (
        (tensile_data_grouped.index.get_level_values('Bolt') == 1) & (tensile_data_grouped.index.get_level_values('Test') == 1) |
        (tensile_data_grouped.index.get_level_values('Bolt') == 2) & (tensile_data_grouped.index.get_level_values('Test') == 2) |
        (tensile_data_grouped.index.get_level_values('Bolt') == 4) & (tensile_data_grouped.index.get_level_values('Test') == 10) |
        (tensile_data_grouped.index.get_level_values('Bolt') == 5) & (tensile_data_grouped.index.get_level_values('Test') == 11) |
        (tensile_data_grouped.index.get_level_values('Bolt') == 7) & (tensile_data_grouped.index.get_level_values('Test') == 11) |
        (tensile_data_grouped.index.get_level_values('Bolt') == 8) & (tensile_data_grouped.index.get_level_values('Test') == 11) |
        (tensile_data_grouped.index.get_level_values('Bolt') == 9) & (tensile_data_grouped.index.get_level_values('Test') == 11)
    )
    tensile_data_grouped['Fracture'] = fracture_conditions.astype(int)

    
    # Reset index to make 'Bolt' and 'Test' columns regular columns
    tensile_data_grouped.reset_index(inplace=True)

    tensile_data_grouped['Bolt'] = pd.to_numeric(tensile_data_grouped['Bolt'], errors='coerce')
    tensile_data_grouped['Test'] = pd.to_numeric(tensile_data_grouped['Test'], errors='coerce')
    inspection_data['Bolt'] = pd.to_numeric(inspection_data['Bolt'], errors='coerce')
    inspection_data['Test'] = pd.to_numeric(inspection_data['Test'], errors='coerce')

    # Merge the dimension inspection and tensile testing dataframes
    merged_df = pd.merge(inspection_data, tensile_data_grouped, on=['Bolt', 'Test'])

    pivot_df = merged_df.pivot_table(index=['Bolt', 'Test'], columns='Name', values='Meas. Value', aggfunc='first').reset_index()

    # Merge pivot_df with the original DataFrame to retain other columns
    merged_df = pd.merge(merged_df[['Bolt', 'Test', 'Max_Load', 'Max_Position', 'Max_Strain', 'Max_Stress', 'Avg_Stress', 'Fracture']], pivot_df, on=['Bolt', 'Test'])

    merged_df = merged_df.drop_duplicates(subset=['Bolt', 'Test'])

    columns = list(merged_df.columns)

    # Remove 'Fracture' column from the list
    columns.remove('Fracture')

    # Append 'Fracture' column at the end of the list
    columns.append('Fracture')

    # Reorder the columns in the DataFrame
    merged_df= merged_df[columns]

    merged_df.reset_index(drop=True, inplace=True)

    # Reassign Fracture column results
    fracture_conditions = ((merged_df['Bolt'] == 1) & (merged_df['Test'] == 1) |
                           (merged_df['Bolt'] == 2) & (merged_df['Test'] == 1) |
                           (merged_df['Bolt'] == 4) & (merged_df['Test'] == 10) |
                           (merged_df['Bolt'] == 5) & (merged_df['Test'] == 11) |
                           (merged_df['Bolt'] == 7) & (merged_df['Test'] == 11) |
                           (merged_df['Bolt'] == 8) & (merged_df['Test'] == 11) |
                           (merged_df['Bolt'] == 9) & (merged_df['Test'] == 11))
    
    merged_df['Fracture'] = fracture_conditions.astype(int)


    ### HANDLE ROWS WITH MISSING VALUES ###
    rows_with_nan = merged_df[merged_df.isna().any(axis=1)]

    # Display rows with NaN values
    # print(rows_with_nan)

    required_columns = merged_df[['Bolt', 'Test']]

    # Identify which columns have missing values for each row
    missing_info = merged_df.isnull().apply(lambda x: ', '.join(merged_df.columns[x]), axis=1)
    merged_df['Missing_Columns'] = missing_info

    # Create a DataFrame with 'Bolt', 'Test', and 'Missing_Columns'
    final_df = merged_df[['Bolt', 'Test', 'Missing_Columns']]

    # Display the final DataFrame
    # print(final_df)

    for index, row in merged_df.iterrows():
        # Check for NaN values in the row
        nan_columns = row[row.isna()].index.tolist()
    
        # Iterate over the columns with NaN values
        for nan_column in nan_columns:
            # Check if there is a corresponding column with similar name but ending with '_90'
            equivalent_column = nan_column + '_90'
            
            # If the equivalent column exists and is not NaN, fill the NaN value with its value
            if equivalent_column in row.index and not pd.isna(row[equivalent_column]):
                merged_df.at[index, nan_column] = row[equivalent_column]

    return merged_df

# Run Descriptive Stats, Correlation Matrix, Outlier Handling
def data_quality(data):

    data = data.drop('Missing_Columns', axis=1)

    descriptive_stats = data.describe(include='all')

    std_row = descriptive_stats.loc['std']

    sorted_by_std_dev = std_row.sort_values(ascending=False)

    # print("Sorted by Standard Deviation:")
    # print(sorted_by_std_dev)

    # Correlation matrix on entire dataset
    correlation_matrix = data.corr()
    # print(correlation_matrix)

    # plt.figure(figsize=(12, 10))
    # plt.tight_layout()
    # sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f")
    # plt.title("Correlation Matrix")
    # plt.show()

    # Correlation Matrix for smaller subset of Features
    subset_df = data[['Bolt', 'Test', 'Max_Load','Minor_OD_3', 'Overall_Length', 'Overall_Length_90', 'Fracture',
                       'Max_Position', 'Angle_Right_1_90', 'Angle_Right_2_90', 'Angle_Right_3_90', 'Pitch_Right_1_90',
                         'Pitch_Right_3_90', 'Angle_Left_3', 'Angle_Left_4', 'Major_OD_4', 'Angle_Left_1', 'Major_OD_2', 'Major_OD_1']]
    
    
    subset_df = subset_df[~subset_df['Bolt'].isin([1, 2])]
    subset_correlation_matrix = subset_df.corr()

    # plt.figure(figsize=(12, 10))
    # plt.tight_layout()
    # sns.heatmap(subset_correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
    # plt.title("Correlation Matrix")
    # plt.show()


    # Evaluate for strong correlations
    strong_correlations = correlation_matrix[(correlation_matrix.abs() > 0.5) & (correlation_matrix.abs() < 1.0)].stack().reset_index()
    strong_correlations.columns = ['Variable 1', 'Variable 2', 'Correlation']
    # print(strong_correlations)
    
    # Z Score calculation
    z_scores = subset_df.apply(zscore)

    # Identify outliers (threshold = 3 standard deviations from the mean)
    outliers = (z_scores > 3) | (z_scores < -3)

    # Define the subset of columns excluding 'Fracture'
    subset_columns = subset_df.columns.difference(['Fracture'])

    # for column in subset_df.columns:
    #     print("Outliers in column '{}':".format(column))
    #     print(subset_df[outliers[column]])

    # #Count number of outliers for each column
    # num_outliers = outliers.sum()
    # print("Number of outliers for each column:")
    # print(num_outliers)

    # exclude_columns = ['Bolt', 'Test', 'Fracture', 'Max_Load', 'Max_Position']

    # # Create a box plot for all columns except the excluded
    # box_plot_data = subset_df.drop(columns=exclude_columns)

    # plt.boxplot(box_plot_data.values, labels=box_plot_data.columns)
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.title('Box Plot for Outliers')
    # plt.xlabel('Feature')
    # plt.ylabel('Measurement')
    # plt.grid(True)

    # plt.show()


    # plt.boxplot(subset_df['Max_Load'], labels=['Max_Load'])
    # plt.tight_layout()
    # plt.title('Box Plot for Outliers')
    # plt.xlabel('Feature')
    # plt.ylabel('Measurement')
    # plt.grid(True)

    # plt.show()

    # Iterate over each column and bolt to replace outliers with the average
    for column in subset_columns:
        for bolt in subset_df['Bolt'].unique():
            # Find previous non-outlier values for the same bolt
            previous_values = subset_df.loc[(subset_df['Bolt'] == bolt) & (~outliers[column]), column]
            # Calculate the average of non-outlier values
            column_avg = previous_values.mean() if not previous_values.empty else np.nan
            # Replace outliers with the calculated average
            subset_df.loc[outliers[column] & (subset_df['Bolt'] == bolt), column] = column_avg

    # subset_df.to_csv('outliershandled122.csv')

    return subset_df

# Increase size of dataframe to include 'n' bolts and 11 max tests
def bootstrap_fixed_structure(dataframe, num_bolts, max_tests_per_bolt):
    # Initialize a list to hold the new bootstrapped data
    new_data = []
    print(len(dataframe))

    # Iterate through the specified number of bolts
    for bolt_id in range(1, num_bolts + 1):
        # Resample the original data with replacement
        sampled_tests = dataframe.sample(n=max_tests_per_bolt, replace=True, random_state=bolt_id).copy()

        # Assign new bolt ID and test numbers
        sampled_tests['Bolt'] = bolt_id
        sampled_tests['Test'] = range(1, len(sampled_tests) + 1)

        # Add the new data to the list
        new_data.append(sampled_tests)

    quality = pd.concat(new_data, ignore_index=True)
    print(len)

    return quality

# Create SQL Database for Azure Digital Twins
def create_SQL(sql_script_path):

    # Command to execute the SQL script
    command = f"sqlcmd -S hame0030-sql-server.database.windows.net -d dsa_5900_sp24 -U hame0030 -P Cougars1 -i {sql_script_path}"

    result = subprocess.run(command, shell=True)

    if result.returncode == 0:
        print('SQL_Table_Created')
    else:
        print('Failed,', result.returncode)

# Load dataframe into SQL Database for Azure Digital Twins
def load_SQL(engine, table_name, df_bootstrapped):

    df_bootstrapped.to_sql(name=table_name, con=engine, if_exists='append', index=False)
    
    result = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(result)

# Split data for testing, Data Evaluation using K Means Clusters and PCA
def split_data(df_bootstrapped):

    # Select only the numeric data for clustering and scaling
    numeric_data = df_bootstrapped.select_dtypes(include=['float64', 'int64'])

    # Standardize the numeric data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Apply PCA to reduce the data for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    #print(scaled_data)

    # Determine the optimal number of clusters using elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve to find the optimal number of clusters
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
    # plt.title('Elbow Method for Optimal Clusters')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Within-Cluster Sum of Squares')
    #plt.show()

    # Perform K-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Add the cluster labels to the DataFrame for future reference
    df_bootstrapped['Cluster'] = cluster_labels

    # Visualize the clusters in the 2D space
    # plt.figure(figsize=(8, 6))
    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    # plt.title('K-means Clustering')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.colorbar(label='Cluster')
    #plt.show()

    # Get the cluster labels
    cluster_labels = kmeans.labels_
    df_bootstrapped['Cluster'] = cluster_labels

    # Group the data by clusters
    cluster_groups = df_bootstrapped.groupby('Cluster')

    # Calculate the mean and standard deviation for each cluster
    cluster_stats = cluster_groups.agg([np.mean, np.std])
    # print(cluster_stats)

    cluster_variance = cluster_groups.var()

    # print("Feature variance across clusters:")
    # print(cluster_variance)

    cluster_counts = df_bootstrapped.groupby('Cluster').size()

    # print("Number of rows within each cluster:")
    # print(cluster_counts)

    cluster_stats.to_csv('Clusters.csv')

    x = df_bootstrapped.drop(['Fracture', 'Cluster', 'Max_Position', 'Max_Load'], axis=1) # Features
    y = df_bootstrapped['Fracture'] # Target variable
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # x_train.to_csv('/Users/chashamel/Documents/DSA5900/x_train.csv')
    # x_test.to_csv('/Users/chashamel/Documents/DSA5900/x_test.csv')
    # y_train.to_csv('/Users/chashamel/Documents/DSA5900/y_train.csv')
    # y_test.to_csv('/Users/chashamel/Documents/DSA5900/y_test.csv')
    
    return x_train, x_test, y_train, y_test, x, y

# Random Forest Training and Prediction
def train_random_forest(x_train, y_train, x_test, y_test, x, y):

    strat_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    random_forest = RandomForestClassifier(class_weight='balanced')
    random_forest.fit(x_train, y_train)  # Train the model
    cv_scores = cross_val_score(random_forest, x_train, y_train, cv=kf, scoring='accuracy')
    cv_scores2 = cross_val_score(random_forest, x_train, y_train, cv=strat_kf, scoring='accuracy')

    print("RF_Cross-Validation Scores - KFolds:", cv_scores)
    print("RF_Mean Accuracy - KFolds:", np.mean(cv_scores)) 

    print("RF_Cross-Validation Scores - Stratified K Folds:", cv_scores2)
    print("RF_Mean Accuracy - Stratified K Folds :", np.mean(cv_scores2)) 

    # Predict on the test set
    y_pred_rf = random_forest.predict(x_test)

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix:\n", conf_matrix)

    # Evaluate the model
    print("Random Forest Executed!")
    print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Extract feature importance
    feature_importance = random_forest.feature_importances_

    total_importance = sum(feature_importance)
    feature_importance_percentage = (feature_importance / total_importance) * 100   

    # Plot feature importance
    feature_names = x.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_percentage})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', orient='h', palette='viridis')
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Feature Importance (%)")
    plt.show()

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print("Random Forest - MSE:", mse_rf, "R-squared:", r2_rf)

    return y_pred_rf

# Decision Tree Training and Prediction
def train_decision_tree(x_train, y_train, x_test, y_test, x, y):

    strat_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)   

    decision_tree = DecisionTreeClassifier(class_weight='balanced')
    decision_tree.fit(x_train, y_train)  # Train the model
    cv_scores = cross_val_score(decision_tree, x_train, y_train, cv=kf, scoring='accuracy')
    cv_scores2 = cross_val_score(decision_tree, x_train, y_train, cv=strat_kf, scoring='accuracy')

    print("DT_Cross-Validation Scores:", cv_scores)
    print("DT_Mean Accuracy:", np.mean(cv_scores)) 
    print("DT_Cross-Validation Scores - Stratified KF:", cv_scores2)
    print("DT_Mean Accuracy - Stratified KF:", np.mean(cv_scores2)) 

    # Predict on the test set
    y_pred_dt = decision_tree.predict(x_test)
    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_dt)
    print("Confusion Matrix:\n", conf_matrix)

    # Evaluate the model
    print("Decision Tree - Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("Classification Report:\n", classification_report(y_test, y_pred_dt))

    # Extract feature importance
    feature_importance = decision_tree.feature_importances_

    total_importance = sum(feature_importance)
    feature_importance_percentage = (feature_importance / total_importance) * 100

    # Plot feature importance
    feature_names = x.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_percentage})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', orient='h', palette='viridis')
    plt.title("Feature Importance in Decision Trees")
    plt.xlabel("Feature Importance (%)")
    plt.show()

    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    print("Decision Tree - MSE:", mse_dt, "R-squared:", r2_dt)
    
    return y_pred_dt

# Logistic Regression Training and Prediction
def train_log_regression(x_train, y_train, x_test, y_test, x, y):

    log_reg = LogisticRegression(max_iter = 10000, class_weight='balanced')

    # Fit the model with the training data
    log_reg.fit(x_train, y_train)

    y_pred_lr = log_reg.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred_lr)
    print("Accuracy:", accuracy)

    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_lr)
    print("Confusion Matrix:\n", conf_matrix)

    # Display classification report
    report = classification_report(y_test, y_pred_lr)
    print("Classification Report:\n", report)

    feature_names = x.columns

    # Get the coefficients from the logistic regression model
    coefficients = log_reg.coef_[0] 

    # Create a DataFrame for feature names and their corresponding coefficients
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Sort the features by their coefficient value
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

    # Plot the feature importance using a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Coefficient'], color='b')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.title('Feature Importance in Logistic Regression')
    plt.xticks(rotation=45)
    plt.show()

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print("Logistic Regression - MSE:", mse_lr, "R-squared:", r2_lr)

# Main function
def main():
    # tensile_folder = "Project_Measurement_Data/2_Tensile_Data"
    # dimensional_folder = "Project_Measurement_Data/1_Model_Data"

    tensile_folder = "/Users/chashamel/Documents/DSA5900/2_Tensile_Data"
    dimensional_folder = "/Users/chashamel/Documents/DSA5900/1_Model_Data"
    
    #### Remove the "" from the headers of tensile data
    #tensile_remove_quotes_from_column_headers(tensile_folder)

    #### Remove headers, change I to i, change 90_ to _90 in file
    #measurement_file_cleanup(dimensional_folder)

    #### Load data into dataframe
    tensile_data, inspection_data = load_data(tensile_folder, dimensional_folder)
    
    # Preprocess Data, Feature Engineering (Max Tensile, Stress Strain), Handle Missing Values, Merge Tensile / Dimensional Data
    data = preprocess_data(tensile_data, inspection_data)

    quality = data_quality(data)

    # Specify the desired number of bolts and tests per bolt
    num_bolts = 100
    max_tests_per_bolt = 11

    # Create a bootstrapped dataset
    df_bootstrapped = bootstrap_fixed_structure(quality, num_bolts, max_tests_per_bolt)

    # ## Push to SQL Database
    # #sql_script_path = "/Users/darren.hamelii/OneDrive - Otter Products/Desktop/DSA5900/Algorithms/DigitalTwinSQL.sql"
    # sql_script_path = "/Users/chashamel/Documents/DSA5900/DigitalTwinSQL.sql"
    # create_SQL(sql_script_path)

    # ## Load SQL DF
    # server = "hame0030-sql-server.database.windows.net"
    # database = "dsa_5900_sp24"
    # username = "hame0030"
    # password = "Cougars1"
    # connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    # engine = sqlalchemy.create_engine(connection_string)
    # table_name = 'digitaltwin'

    # load_SQL(engine, table_name, df_bootstrapped)

    # # Split data into train and test sets
    x_train, x_test, y_train, y_test, x, y = split_data(df_bootstrapped)
    
    # # Train Random Forest model
    train_random_forest(x_train, y_train, x_test, y_test, x, y)
    
    # # Train Decision Tree Model
    train_decision_tree(x_train, y_train, x_test, y_test, x, y)

    # # Train Logistic Regression
    #train_log_regression(x_train, y_train, x_test, y_test, x, y)

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.model_selection import train_test_split


def find_outliers_iqr(data: pd.DataFrame) -> pd.Series:
    """BoxPlot Method:
    Identify outliers in each column of the DataFrame using the IQR method.
    Args:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.Series: A series with the count of outliers for each column.
    """
    if data is None or data.empty:
        raise ValueError("Input data is None or empty.")
    
    if 'Appliances' not in data.columns:
        raise ValueError("Expected 'Appliances' column in the data.")
    
    # Calculate the first quartile (Q1) and third quartile (Q3) for each column
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    # Calculate the interquartile range (IQR) for each column
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers for each column
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Check for outliers in each column and count the number of outliers
    outliers_count = (data < lower_bound) | (data > upper_bound)
    num_outliers = outliers_count.sum()

    return num_outliers




def cleaning_data(data: pd.DataFrame, column_mapping:dict=None, desired_order:list=None) -> pd.DataFrame:
    
    num_columns = int(data.shape[1])

        
    if column_mapping is None:
        print(f"column_mapping: {column_mapping}, rn: {rn}")
        raise ValueError("column_mapping should not be None if rn")

    # Renaming the columns based on the provided mapping. It is important for making the column names more descriptive.
    # Filtering the data to include only the desired columns. it helps to focus on relevant features for analysis.
    data = data.rename(columns=column_mapping) 
    data = data[[col for col in desired_order if col in data.columns]] if desired_order else data
    return data


def preprocess(dataset_path:str, column_mapping: dict = None, desired_order:list=None, forproduction: bool = False, skt: float = 0.5, tsr:int=42, ts:float=0.3 ) -> dict | tuple:
    """
    Preprocess the input DataFrame by renaming columns, handling skewness, scaling features,
    and creating polynomial features. It also splits the data into training and testing sets.
    Args:
        data (pd.DataFrame): Input DataFrame to preprocess.
        column_mapping (dict): A dictionary mapping old column names to new column names.
        desired_order (list): A list of colum names With the correct order 
        forproduction (bool): Flag indicating if the preprocessing is for production use. Defaults to False.
        skt (Float): setting up the threshold. it helps to classify features based on skewness
        tsr (Int): training split random state
        ts: (Float): Test_size, Split the data into training and testing sets.
    Returns:
        dict: A dictionary containing training and testing sets with polynomial features.
        dict structure:
            if forproduction is False:
                {
                    "Training": (X_train_poly, X_test_poly, y_train, y_test),
                    "Full": (X, y)
                }
            else: (X, y)
                
    """


    data_raw = pd.read_csv(dataset_path)
    data = data_raw.copy()
    data = cleaning_data(data, column_mapping, desired_order)
    
    # Ensure 'Appliances' column exists. because it's used later as target variable
    if 'Appliances' not in data.columns:
        raise ValueError("Expected 'Appliances' column in the data for target variable.")
    


    

    #assigning new_data as new name of dataframe. It's optional for clarity.
    # new_data = data.copy()


    #examining the skewness in the dataset to check the distribution. important for transformation
    # it helps to identify which features need transformation to approximate normality.
    # 'Appliances' column is excluded as it is the target variable.
    numeric_data = data.select_dtypes(include='number').drop(columns=['Appliances'])
    skewness = numeric_data.skew()
    
    #ginding the absolute value. it helps to compare the skewness regardless of direction
    # it reterns the absolute skewness values for easier comparison
    skewness = abs(skewness)
    
    # setting up the threshold. it helps to classify features based on skewness
    skewness_threshold = skt
    
    # Separate features into symmetrical and skewed based on skewness threshold. it helps to identify which features need transformation
    symmetrical_features = skewness[abs(skewness) < skewness_threshold].index
    skewed_features = skewness[abs(skewness) >= skewness_threshold].index
    
    # Create new DataFrames for symmetrical and skewed features. it helps to organize features based on their distribution
    print('FEATURES FOLLOWED SYMMETRICAL DISTRIBUTION :')
    symmetrical_data = data[symmetrical_features]
    print(symmetrical_features)
    
    print('FEATURES FOLLOWED SKEWED DISTRIBUTION :')
    skewed_data = data[skewed_features]
    print(skewed_features)

    
    
    # Initialize the PowerTransformer. it helps to reduce skewness and approximate normality
    # this is particularly useful for features that do not follow a normal distribution.
    power_transformer = PowerTransformer()
    
    # Fit and transform the data using the PowerTransformer
    power_transformed = pd.DataFrame(
        power_transformer.fit_transform(skewed_data),
        columns=skewed_data.columns,
        index=skewed_data.index
    )
    power_transformed.columns = skewed_data.columns

    from sklearn.preprocessing import StandardScaler
    
    # StandardScaler to scale the features to have mean 0 and variance 1. it is important for PCA and many machine learning algorithms.
    # it ensures that all features contribute equally to the analysis. 
    # it returns the scaled features with standardized values.
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(power_transformed),
        columns=power_transformed.columns,
        index=power_transformed.index
    )
    scaled_data.columns = power_transformed.columns

    # Combine the scaled skewed features with the symmetrical features. it creates a unified dataset for modeling.
    processed_data = pd.concat([scaled_data, symmetrical_data], axis=1)

    #assigning the independent and dependent feature. it is important for model training and evaluation.
    X = processed_data
    y = data['Appliances']

    # --- original snippet ---
    
    # Split the data into training and testing sets. it is important for evaluating model performance on unseen data.
    if not forproduction:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=tsr)
    
    # Specify the degree of polynomial (you can change this based on your data). it helps to capture non-linear relationships between features and target variable.
    # the value of degree can be adjusted based on the complexity of the data. like 2 or 3.
    degree = 2
    
    # Create polynomial features for the training and testing sets. it expands the feature set to include polynomial combinations of the original features.
    # it is useful for models that can benefit from non-linear relationships.
##### --------------------------------Hier gebleven ( 30 10 2025; 15:30 )----------------------------------------------- #### 
    if not forproduction:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)

    return {
        "Training": (X_train_poly, X_test_poly, y_train, y_test),
        "Full": (X, y)
    } if not forproduction else (X, y)



def save_train_test_data(path, X_train, X_test, y_train, y_test):
    """Save the training and testing data to CSV files.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    """
    if not path:
        raise ValueError("Path must be provided to save the data.")
    elif X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Training and testing data must not be None.")
    

    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train).reset_index(drop=True)
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)



    X_train_df.to_csv(f"{path}/X_train.csv", index=False)
    X_test_df.to_csv(f"{path}/X_test.csv", index=False)
    y_train_df.to_csv(f"{path}/y_train.csv", index=False)
    y_test_df.to_csv(f"{path}/y_test.csv", index=False)


"""
This model is designed to predict appliance energy consumption based on various temperature and humidity features from different rooms and outside conditions. The model uses polynomial features to capture non-linear relationships in the data. The predictions are made on a sample of the input data, and the results are printed for review.
Created by: Nova Energy Research & Development
in Oktober, 2025

"""
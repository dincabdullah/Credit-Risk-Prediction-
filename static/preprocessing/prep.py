import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from static.classifiermodels.classifiers import get_all_trained_models



def load_application_train():
    data = pd.read_csv("/Users/dincabdullah/Downloads/datawarehouse 2/creditriskpredictionapp/static/preprocessing/german_credit_data.csv")
    return data


def add_new_data(new_data):
    df1 = load_application_train()
    dict_row = {
        "Unnamed: 0" : 1000,
        "Age" : int(new_data["age"]),
        "Sex" : new_data["gender"],
        "Job" : int(new_data["job"]),
        "Housing" : new_data["housing"],
        "Saving accounts" : new_data["savingaccounts"],
        "Checking account" : new_data["checkingaccounts"],
        "Credit amount" : int(new_data["creditamount"]),
        "Duration" : int(new_data["duration"]),
        "Purpose" : new_data["purpose"],
        "Risk" : "good" 
        }
    df2 = pd.DataFrame(dict_row, index=[0])


    df = pd.concat([df1, df2], ignore_index=True)
    return df


def make_preprocessing(request):
    df = add_new_data(request.POST)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    Cat_Age = []
    for i in df["Age"]:
        if i < 25:
            Cat_Age.append("Young")
        elif (i >= 25) and (i < 40):
            Cat_Age.append("Adult")
        elif (i >= 40) and (i < 65):
            Cat_Age.append("Middle-Age")
        elif i >= 65:
            Cat_Age.append("Old")

    df["Cat Age"] = Cat_Age
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    # for col in num_cols:
    #     target_summary_with_num(df, "Risk", col)

    df['Risk'] = df['Risk'].replace({'good': 0, 'bad': 1}, regex=True)

    for col in cat_cols:
        target_summary_with_cat(df, "Risk", col)

    for col in num_cols:
        print(col, check_outlier(df, col))

    for col in num_cols:
        if check_outlier(df, col):
            replace_with_thresholds(df, col, 0.1, 0.9)

    for col in num_cols:
        print(col, check_outlier(df, col))

    check_outlier(df, col)

    df.isnull().values.any()
    df.isnull().sum()
    df.isnull().sum().sum()

    na_cols = missing_values_table(df, True)
    missing_vs_target(df, "Risk", na_cols)
    df.drop("Checking account", axis=1, inplace=True)
    df["Saving accounts"] = df["Saving accounts"].fillna(df["Saving accounts"].mode().iloc[0])
    na_cols = missing_values_table(df, True)
    df.isnull().values.any()
    df[num_cols].corr()
    high_correlated_cols(df[num_cols])


    df["NEW_monthly_repayment"] = df["Credit amount"] / df["Duration"]

    df["NEW_Age*Job"] = df["Age"] * df["Job"]

    df['NEW_Housing*Job'] = df['Housing'].map({"free": 1, "rent": 2, "own": 3}) * df['Job']
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]
    print(binary_cols)
    #binary_cols.remove("Risk")

    for col in binary_cols:
        label_encoder(df, col)

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    print(ohe_cols)

    df = one_hot_encoder(df, ohe_cols, drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    print(num_cols)

    scale = StandardScaler()
    df[num_cols] = scale.fit_transform(df[num_cols])
    print(df)
    return df


def execute_order_by(request):
    print("Dict of Request: ", request.POST)
    df = make_preprocessing(request)
    processed_version_of_the_added_data = df.iloc[- 1]
    credit_risk_result = make_test_given_row(processed_version_of_the_added_data)
    return credit_risk_result


def make_test_given_row(test_data):
    all_classifiers = get_all_trained_models()
    decision_tree = all_classifiers["svm"]

    columns = [
        'Age', 'Sex', 'Credit amount', 'Duration', 'NEW_monthly_repayment',
        'NEW_Age*Job', 'Job_1', 'Job_2', 'Job_3',
        'Housing_own', 'Housing_rent', 'Saving accounts_moderate',
        'Saving accounts_quite rich', 'Saving accounts_rich', 'Purpose_car',
        'Purpose_domestic appliances', 'Purpose_education', 'Purpose_furniture/equipment',
        'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others',
        'Cat Age_Middle-Age', 'Cat Age_Old', 'Cat Age_Young',
        'NEW_Housing*Job_1', 'NEW_Housing*Job_2', 'NEW_Housing*Job_3',
        'NEW_Housing*Job_4', 'NEW_Housing*Job_6', 'NEW_Housing*Job_9'
    ]

    test_rows = []
    test_row = []
    if len(test_data) != 30:
        for key in test_data.keys():
            if key in columns:
                test_row.append(test_data[key])
        test_rows.append(test_row)

    
    print(test_rows[0])
    prediction = decision_tree.predict(test_rows)
    print("Prediction:", prediction)
    return prediction[0]


def just_one_times_read():
    with open("control.txt", "r") as fileControl:
        row = fileControl.readline()
        if row != "done":
            print("not done")
            data = pd.read_csv("/Users/dincabdullah/Downloads/datawarehouse 2/creditriskpredictionapp/static/preprocessing/german_credit_data/german_credit_data.csv")
            with open("control.txt", "w") as fileControl:
                fileControl.write("done")
        else:
            print("marked as done.")


def grab_col_names(dataframe, categorical_threshold=10, cardinal_threshold=20):

    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    numeric_looking_but_categorical = [col for col in dataframe.columns if dataframe[col].dtype != "O" and dataframe[col].nunique() < categorical_threshold]

    categorical_looking_but_cardinal = [col for col in dataframe.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() > cardinal_threshold]

    categorical_cols = categorical_cols + numeric_looking_but_categorical

    categorical_cols = [col for col in categorical_cols if col not in categorical_looking_but_cardinal]

    numeric_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]

    numeric_cols = [col for col in numeric_cols if col not in numeric_looking_but_categorical]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorical Columns: {len(categorical_cols)}")
    print(f"Numeric Columns: {len(numeric_cols)}")
    print(f"Categorical Looking but Cardinal: {len(categorical_looking_but_cardinal)}")
    print(f"Numeric Looking but Categorical: {len(numeric_looking_but_categorical)}")
    return categorical_cols, numeric_cols, categorical_looking_but_cardinal

def target_summary_with_num(dataframe, target, col_name):
    print(dataframe.groupby(target).agg({col_name: "mean"}))

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name, 0.1, 0.9)

    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False
    

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquartile = quartile3 - quartile1

    up = quartile3 + 1.5 * interquartile
    low = quartile1 - 1.5 * interquartile
    return low, up


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}))


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show(block=True)
    return drop_list


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
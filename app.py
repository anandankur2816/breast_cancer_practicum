import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
# loading in the model to predict on the data
pickle_in = open('final_model.sav', 'rb')
classifier = pickle.load(pickle_in)



def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(input_df):
    prediction = classifier.predict(input_df.iloc[:1])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Breast Cancer Prediction")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Streamlit Breast Cancer Classifier ML App </h1>
	</div>
	"""

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)
    df.drop(df.columns[[0, -1]], axis=1, inplace=True)
    # Adding plot
    # I visualized target data in the dataset.
    fig = plt.figure(figsize=(40, 20))
    sns.heatmap(df.corr(), annot=True, cmap='Accent_r')
    st.write("Correlation between features")
    st.write(fig)

    # Displaying the dataset
    st.write("Dataset")
    st.write(df.corr())
    df[["diagnosis"]].value_counts().plot()
    # %%
    data = df
    data = data.rename(columns={"diagnosis": "target"})
    # 1 : M and 0 : B
    data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]
    corr_matrix = data.corr()
    threshold = 0.75
    filter = np.abs(corr_matrix["target"] > threshold)
    corr_features = corr_matrix.columns[filter].tolist()
    fig = plt.figure(figsize=(40, 20))
    sns.clustermap(data[corr_features].corr(), annot=True, fmt=".2f")
    st.write("Correlation Between Features with Corr Threshold 0.75")
    # Displaying an image
    image = Image.open('corelation_75.png')

    st.image(image, caption='Correlation Between Features with Corr Threshold 0.75 of target')

    # Box Plot
    data_melt = pd.melt(data, id_vars="target",
                        var_name="features",
                        value_name="value")

    fig = plt.figure(figsize=(17, 7))
    sns.boxplot(x="features", y="value", hue="target", data=data_melt)
    plt.xticks(rotation=90)
    st.write("Box Plot")
    st.write(fig)

    # Pair plot
    fig = plt.figure(figsize=(17, 7))
    sns.pairplot(data[corr_features], diag_kind="kde", markers="+", hue="target")
    st.write("Pair Plot with a correlation greater than 0.75")
    image = Image.open('pair_plotoutput.png')
    st.image(image, caption='Correlation Between Features with Corr Threshold 0.75')

    # Displaying the Dataset
    st.write("Dataset")
    st.write(df)

    # Creating a sidebar
    st.sidebar.header('User Input Features')
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            radius_mean = st.sidebar.slider('radius_mean', 6.981, 28.11, 14.13)
            texture_mean = st.sidebar.slider('texture_mean', 9.71, 39.28, 19.29)
            perimeter_mean = st.sidebar.slider('perimeter_mean', 43.79, 188.5, 91.97)
            area_mean = st.sidebar.slider('area_mean', 143.5, 2501.0, 654.89)
            smoothness_mean = st.sidebar.slider('smoothness_mean', 0.053, 0.163, 0.096)
            compactness_mean = st.sidebar.slider('compactness_mean', 0.019, 0.345, 0.104)
            concavity_mean = st.sidebar.slider('concavity_mean', 0.0, 0.427, 0.088)
            concave_points_mean = st.sidebar.slider('concave_points_mean', 0.0, 0.201, 0.05)
            symmetry_mean = st.sidebar.slider('symmetry_mean', 0.106, 0.304, 0.181)
            fractal_dimension_mean = st.sidebar.slider('fractal_dimension_mean', 0.05, 0.097, 0.062)
            radius_se = st.sidebar.slider('radius_se', 0.112, 2.873, 0.405)
            texture_se = st.sidebar.slider('texture_se', 0.36, 4.885, 1.216)
            perimeter_se = st.sidebar.slider('perimeter_se', 0.757, 21.98, 2.866)
            area_se = st.sidebar.slider('area_se', 6.802, 542.2, 40.32)
            smoothness_se = st.sidebar.slider('smoothness_se', 0.002, 0.031, 0.007)
            compactness_se = st.sidebar.slider('compactness_se', 0.002, 0.135, 0.025)
            concavity_se = st.sidebar.slider('concavity_se', 0.0, 0.396, 0.032)
            concave_points_se = st.sidebar.slider('concave_points_se', 0.0, 0.053, 0.01)
            symmetry_se = st.sidebar.slider('symmetry_se', 0.008, 0.079, 0.018)
            fractal_dimension_se = st.sidebar.slider('fractal_dimension_se', 0.001, 0.03, 0.004)
            radius_worst = st.sidebar.slider('radius_worst', 7.93, 36.04, 16.27)
            texture_worst = st.sidebar.slider('texture_worst', 12.02, 49.54, 25.68)
            perimeter_worst = st.sidebar.slider('perimeter_worst', 50.41, 251.2, 107.26)
            area_worst = st.sidebar.slider('area_worst', 185.2, 4254.0, 880.58)
            smoothness_worst = st.sidebar.slider('smoothness_worst', 0.071, 0.223, 0.132)
            compactness_worst = st.sidebar.slider('compactness_worst', 0.027, 1.058, 0.254)
            concavity_worst = st.sidebar.slider('concavity_worst', 0.0, 1.252, 0.271)
            concave_points_worst = st.sidebar.slider('concave_points_worst', 0.0, 0.291, 0.115)
            symmetry_worst = st.sidebar.slider('symmetry_worst', 0.156, 0.664, 0.289)
            fractal_dimension_worst = st.sidebar.slider('fractal_dimension_worst', 0.055, 0.208, 0.083)
            data = {'radius_mean': radius_mean,
                    'texture_mean': texture_mean,
                    'perimeter_mean': perimeter_mean,
                    'area_mean': area_mean,
                    'smoothness_mean': smoothness_mean,
                    'compactness_mean': compactness_mean,
                    'concavity_mean': concavity_mean,
                    'concave_points_mean': concave_points_mean,
                    'symmetry_mean': symmetry_mean,
                    'fractal_dimension_mean': fractal_dimension_mean,
                    'radius_se': radius_se,
                    'texture_se': texture_se,
                    'perimeter_se': perimeter_se,
                    'area_se': area_se,
                    'smoothness_se': smoothness_se,
                    'compactness_se': compactness_se,
                    'concavity_se': concavity_se,
                    'concave_points_se': concave_points_se,
                    'symmetry_se': symmetry_se,
                    'fractal_dimension_se': fractal_dimension_se,
                    'radius_worst': radius_worst,
                    'texture_worst': texture_worst,
                    'perimeter_worst': perimeter_worst,
                    'area_worst': area_worst,
                    'smoothness_worst': smoothness_worst,
                    'compactness_worst': compactness_worst,
                    'concavity_worst': concavity_worst,
                    'concave_points_worst': concave_points_worst,
                    'symmetry_worst': symmetry_worst,
                    'fractal_dimension_worst': fractal_dimension_worst}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    # radius_mean = st.text_input("Radius Mean", "")
    # texture_mean = st.text_input("Text", "")
    # perimeter_mean = st.text_input("perimeter_mean", "")
    # area_mean = st.text_input("area_mean", "")
    # smoothness_mean = st.text_input("smoothness_mean", "")
    # compactness_mean = st.text_input("compactness_mean", "")
    # concavity_mean = st.text_input("concavity_mean", "")
    # concave_points_mean = st.text_input("concave_points_mean", "")
    # symmetry_mean = st.text_input("symmetry_mean", "")
    # fractal_dimension_mean = st.text_input("fractal_dimension_mean", "")
    # radius_se = st.text_input("radius_se", "")
    # texture_se = st.text_input("texture_se", "")
    # perimeter_se = st.text_input("perimeter_se", "")
    # area_se = st.text_input("area_se", "")
    # smoothness_se = st.text_input("smoothness_se", "")
    # compactness_se = st.text_input("compactness_se", "")
    # concavity_se = st.text_input("concavity_se", "")
    # concave_points_se = st.text_input("concave_points_se", "")
    # symmetry_se = st.text_input("symmetry_se", "")
    # fractal_dimension_se = st.text_input("fractal_dimension_se", "")
    # radius_worst = st.text_input("radius_worst", "")
    # texture_worst = st.text_input("texture_worst", "")
    # perimeter_worst = st.text_input("perimeter_worst", "")
    # area_worst = st.text_input("area_worst", "")
    # smoothness_worst = st.text_input("smoothness_worst", "")
    # compactness_worst = st.text_input("compactness_worst", "")
    # concavity_worst = st.text_input("concavity_worst", "")
    # concave_points_worst = st.text_input("concave_points_worst", "")
    # symmetry_worst = st.text_input("symmetry_worst", "")
    # fractal_dimension_worst = st.text_input("fractal_dimension_worst", "")

    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(input_df)
    if result == 0:
        st.success('The Breast Cancer is Benign')
    else:
        st.success('The Breast Cancer is Malignant')


if __name__ == '__main__':
    main()

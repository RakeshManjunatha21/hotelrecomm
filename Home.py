import streamlit as st
import time
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import accuracy_score
import base64
import hmac

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('staticfiles\\Img\\login_page.jpg')

#Password code ----------------------------
def check_password():

    def login_form():
        """Welcome to AMIVESTOR: Restaurant Market Analysis\n
        Please enter your credentials below to access the system."""
        
        # Set form background color to gray
        st.markdown(
            """
            <style>
            .st-cc {
                background-color: #808080; /* Gray background color */
                padding: 20px;
                border-radius: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # Set text color and shadow
        st.markdown(
            """
            <style>
            .text-shadow {
                text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # text enter cred 
        st.markdown(
            """
            <style>
            .text-shadow-inside {
                text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        #Transparent gray background color
        st.markdown(
            """
            <style>
            .st-cc-trans {
                background-color: rgba(128, 128, 128, 0.5); /* Transparent gray background color */
                padding: 20px;
                border-radius: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.form("🔐 Log In",clear_on_submit=True,border=False):
            # Apply text color to the form title
            # st.markdown("<h2 class='text-shadow st-cc'>Welcome to Restaurant Market Analysis</h2>", unsafe_allow_html=True)
            # st.markdown("<p class='text-shadow st-cc'>AI Based Prime Hotel Recommendations & Feasibility Analysis</p>", unsafe_allow_html=True)
            # st.markdown("<p class='text-shadow st-cc'>Please enter your credentials below to access the system.</p>", unsafe_allow_html=True)
            st.markdown("<h2 class='text-shadow st-cc-trans'>Welcome to AMIVESTOR: Restaurant Market Analysis<br> <p>AI Based Prime Hotel Recommendations & Feasibility Analysis</p></h2>", unsafe_allow_html=True)
            st.markdown("<p class='text-shadow-inside'>Please enter your credentials below to access the system.</p>", unsafe_allow_html=True)

            # st.text_input("👤 Username", key="username")
            st.text_input("👤 :green[Username]", key="username")
            st.text_input("🔑 :green[Password]", type="password", key="password")
            st.form_submit_button("🚀 Log In",on_click=password_entered)


    def password_entered():
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    login_form()
    if "password_correct" in st.session_state:
        st.error("""
                 Invalid Credentials!\n
                 For assistance, kindly reach out to amivestor@assist.ai
                 """)
    return False


if not check_password():
    st.stop()


#ML Code -----------------------------------------------------------------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .st-cc-trans {
        background-color: rgba(128, 128, 128, 0.5); /* Transparent gray background color */
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .text-shadow {
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('staticfiles\\Img\\edit1.png')
# sidebar_bg('bgnew4.jpeg')

st.session_state.page = "Home"

# st.markdown("<h2 class='text-shadow st-cc-trans'>Welcome to Restaurant Market Analysis<br> <p>AI Based Prime Hotel Recommendations & Feasibility Analysis</p></h2>", unsafe_allow_html=True)
title = "Forecasting Success"
title_desc = "Analyzing Market Trends For New Restaurant Ventures"

# st.title(title)
st.markdown("<h3 class='text-shadow st-cc-trans'>Analyzing Market Trends For New Restaurant Ventures</h3><br><br>", unsafe_allow_html=True)
# st.subheader("""What do you want us to predict for you?""")


# Function to perform prediction based on selected factors
def predict_success_probability(area_code, average_cost, food_rating, service_rating, look_and_feel_rating, al_num, selected_cuisines):
    # Placeholder for prediction logic
    return 45.4555  # Dummy value for demonstration

# Apply CSS for transparent background to radio button options
st.markdown(
    """
    <style>
    .st-trans-radio .stRadioButton > div:first-child {
        background-color: rgba(128, 128, 128, 0.5); /* Transparent gray background color */
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the form sections based on selected location
# selected_location = st.radio("What do you want us to predict for you?", ["Feasible Location", "Success Probability"], key='prediction_radio')
with st.markdown('<div class="st-trans-radio">', unsafe_allow_html=True):
    selected_location = st.radio(
        "What do you want us to predict for you?",
        ["Feasible Location", "Success Probability"],
        key='prediction_radio'
    )

if selected_location == "Success Probability":
    st.subheader("Factors for Success Probability")
    Latitude = st.text_input("Enter the Latitude: ")
    Longitude = st.text_input("Enter the Longitude: ")
    area_code = st.text_input("Area Code", "")
    alcohol_option = st.radio("Alcohol Avaliablity ", ["Yes", "No"])
    veg_nonveg = st.radio("Veg/Non-Veg Avaliablity", ["Veg", "Non-Veg", "Both"])
    al_num = 1 if alcohol_option == "Yes" else 0
    veg_nonveg_num = 1 if veg_nonveg == "Non-Veg" else 0

    cuisines_options = ['Italian', 'North Indian', 'Chinese', 'Continental', 'European', 'Mediterranean', 'Andhra', 'Mughlai', 'Asian', 'Cafe', 'Thai', 'Pizza', 'Bakery', 'Biryani', 'Street Food', 'Fast Food', 'American', 'Seafood', 'Gujarati', 'Rajasthani', 'Burger', 'Maharashtrian', 'Beverages', 'Tibetan', 'Mangalorean', 'Arabian', 'Mithai', 'Sandwich', 'Salad', 'Kerala', 'Desserts', 'Finger Food', 'Indian', 'French', 'Healthy Food', 'Juices', 'Mexican', 'Irish', 'Ice Cream']
    selected_cuisines = st.multiselect("Cuisines", cuisines_options)
    
    print(selected_cuisines)
    print("-----------------")
    # Button to trigger the prediction
    if st.button("Predict Rating"):
        with st.spinner('Prediction...'):
            try:
                #Comment-1
                print("len(selected_cuisines):",len(selected_cuisines))
                # loaded_rf_model_success = joblib.load('rf_model_success.joblib')
                # loaded_scaler = joblib.load('scaler_success.joblib')
                #Comment -2

                df = pd.read_excel("Bangalore Dataset Manual.xlsx")
                df['Combined_Variable'] = df['Food rating'] + df['Service rating'] + df['Look and Feel rating']
                df['Success Probability'] = (df['Combined_Variable'] - (3 * 1)) / (3 * (5 - 1))
                Cuisines_count = []
                for i in df.index:
                    cnt = df['Italian'][i] + df['North Indian'][i] + df['Chinese'][i] + df['Continental'][i] + df['European'][i] + df['Mediteranean'][i] + df['Andhra'][i] + df['Mughlai'][i] + df['Asian'][i] + df['Cafe'][i] + df['Thai'][i] + df['Pizza'][i] + df['Bakery'][i] + df['Biryani'][i] + df['Street Food'][i] + df['Fast Food'][i] + df['American'][i] + df['Sea Food'][i] + df['Gujrati'][i] + df['Rajasthani'][i] + df['Burger'][i] + df['Maharasthrian'][i] + df['Beverages'][i] + df['Tibetian'][i] + df['Mangalorean'][i] + df['Arabian'][i] + df['Mithai'][i] + df['Sandwich'][i] + df['Salad'][i] + df['Kerala'][i] + df['Deserts'][i] + df['Finger Food'][i] + df['Indian'][i] + df['French'][i] + df['Healthy Food'][i] + df['Juices'][i] + df['Mexican'][i] + df['Irish'][i] + df['Ice Cream'][i]
                    Cuisines_count.append(cnt)
                df['Cuisines_count'] = Cuisines_count
                selected_columns = ['Latitude', 'Longitude', 'Alcohol availability(Yes-1/No-0)', 'Veg(0)/Non-veg(1)', 'Cuisines_count', 'Success Probability']
                df_selected = df[selected_columns]
                df_selected['Alcohol availability(Yes-1/No-0)'] = df_selected['Alcohol availability(Yes-1/No-0)'].fillna(0).astype(int)
                df_selected.dropna(inplace=True)
                X = df_selected.drop(['Success Probability'], axis=1)
                y_success_probability = df_selected['Success Probability']
                X_train, X_test, y_train_success, y_test_success = train_test_split(
                                    X, y_success_probability, test_size=0.2, random_state=42
                                )
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                rf_model_success = RandomForestRegressor(random_state=42)
                rf_model_success.fit(X_train_scaled, y_train_success)
                y_pred_success = rf_model_success.predict(X_test_scaled)
                mse_success = mean_squared_error(y_test_success, y_pred_success)

                #PREDICT
                test_case = pd.DataFrame({
                'Latitude': [float(Latitude)],
                'Longitude': [float(Longitude)],
                'Alcohol availability(Yes-1/No-0)': [int(al_num)],
                'Veg(0)/Non-veg(1)': [int(veg_nonveg_num)],  # 1 for Non-veg, 0 for Veg
                'Cuisines_count':[int(len(selected_cuisines))]
                })
                test_case = pd.get_dummies(test_case)
                test_case_scaled = scaler.transform(test_case)
                success_probability_prediction = rf_model_success.predict(test_case_scaled)
                txt = "Success Prediction Percentage: "+ str(round(success_probability_prediction[0]*100,2))
                htmlstr1=f"""<p style='background-color:green;
                                                        color:white;
                                                        font-size:18px;
                                                        border-radius:3px;
                                                        line-height:60px;
                                                        padding-left:17px;
                                                        opacity:0.6'>
                                                        {txt}</style>
                                                        <br></p>""" 
                st.markdown(htmlstr1,unsafe_allow_html=True)
            except:
                txt = "Invalid Input!!!!"
                htmlstr1=f"""<p style='background-color:red;
                                                        color:white;
                                                        font-size:18px;
                                                        border-radius:3px;
                                                        line-height:60px;
                                                        padding-left:17px;
                                                        opacity:0.6'>
                                                        {txt}</style>
                                                        <br></p>""" 
                st.markdown(htmlstr1,unsafe_allow_html=True)


elif selected_location == "Feasible Location":
    st.subheader("Feasible Location Analysis")
    step_values = np.arange(1, 5.1, 0.1)
    AverageCost = st.text_input("Enter the estimated average cost for 2 members: ")
    ZomatoRating = st.selectbox('Select the Zomato Rating', step_values, format_func=lambda x: f"{x:.1f}")
    no_of_review = st.text_input("Number of Review: ")
    food_rating = st.selectbox('Select the Food Rating', step_values, format_func=lambda x: f"{x:.1f}")
    service_rating = st.selectbox('Select the Service Rating', step_values, format_func=lambda x: f"{x:.1f}")
    looknfeel_rating = st.selectbox('Select the look and feel Rating', step_values, format_func=lambda x: f"{x:.1f}")
    alcohol_option = st.radio("Alcohol Avaliablity ", ["Yes", "No"])
    veg_nonveg = st.radio("Veg/Non-Veg Avaliablity", ["Veg", "Non-Veg", "Both"])
    cuisines_options = ['Italian', 'North Indian', 'Chinese', 'Continental', 'European', 'Mediterranean', 'Andhra', 'Mughlai', 'Asian', 'Cafe', 'Thai', 'Pizza', 'Bakery', 'Biryani', 'Street Food', 'Fast Food', 'American', 'Seafood', 'Gujarati', 'Rajasthani', 'Burger', 'Maharashtrian', 'Beverages', 'Tibetan', 'Mangalorean', 'Arabian', 'Mithai', 'Sandwich', 'Salad', 'Kerala', 'Desserts', 'Finger Food', 'Indian', 'French', 'Healthy Food', 'Juices', 'Mexican', 'Irish', 'Ice Cream']
    selected_cuisines = st.multiselect("Cuisines", cuisines_options)
    al_num = 1 if alcohol_option == "Yes" else 0
    veg_nonveg_num = 1 if veg_nonveg == "Non-Veg" else 0

    if st.button("Predict Location"):
        with st.spinner('Prediction...'):
            # loaded_rf_model_lat = joblib.load('Models\\Location Prediction\\Latitude\\rf_model_latitude.joblib')
            # loaded_scaler = joblib.load('Models\\Location Prediction\\Latitude\\scaler_latitude.joblib')
            try:
                test_case = pd.DataFrame({
                'Average Cost (for 2)': [float(AverageCost)],
                'Zomato Rating(out of 5)': [float(ZomatoRating)],
                'No of reviews': [int(no_of_review)],
                'Food rating': [float(food_rating)],
                'Service rating': [float(service_rating)],
                'Look and Feel rating': [float(looknfeel_rating)],
                'Alcohol availability(Yes-1/No-0)': [int(al_num)],
                'Veg(0)/Non-veg(1)': [int(veg_nonveg_num)],
                'Cuisines_count':[int(len(selected_cuisines))]
                })
            
            
            

                df = pd.read_excel("Bangalore Dataset Manual.xlsx")
                df['Combined_Variable'] = df['Food rating'] + df['Service rating'] + df['Look and Feel rating']
                df['Success Probability'] = (df['Combined_Variable'] - (3 * 1)) / (3 * (5 - 1))
                Cuisines_count = []
                for i in df.index:
                    cnt = df['Italian'][i] + df['North Indian'][i] + df['Chinese'][i] + df['Continental'][i] + df['European'][i] + df['Mediteranean'][i] + df['Andhra'][i] + df['Mughlai'][i] + df['Asian'][i] + df['Cafe'][i] + df['Thai'][i] + df['Pizza'][i] + df['Bakery'][i] + df['Biryani'][i] + df['Street Food'][i] + df['Fast Food'][i] + df['American'][i] + df['Sea Food'][i] + df['Gujrati'][i] + df['Rajasthani'][i] + df['Burger'][i] + df['Maharasthrian'][i] + df['Beverages'][i] + df['Tibetian'][i] + df['Mangalorean'][i] + df['Arabian'][i] + df['Mithai'][i] + df['Sandwich'][i] + df['Salad'][i] + df['Kerala'][i] + df['Deserts'][i] + df['Finger Food'][i] + df['Indian'][i] + df['French'][i] + df['Healthy Food'][i] + df['Juices'][i] + df['Mexican'][i] + df['Irish'][i] + df['Ice Cream'][i]
                    Cuisines_count.append(cnt)
                df['Cuisines_count'] = Cuisines_count
                selected_columns2 = ['Latitude','Longitude','Average Cost (for 2)', 'Zomato Rating(out of 5)','No of reviews', 'Food rating', 'Service rating',
                                'Look and Feel rating', 'Alcohol availability(Yes-1/No-0)',
                                'Veg(0)/Non-veg(1)','Cuisines_count']
                df_selected2 = df[selected_columns2]
                df_selected2['Alcohol availability(Yes-1/No-0)'] = df_selected2['Alcohol availability(Yes-1/No-0)'].fillna(0).astype(int)
                df_selected2.dropna(inplace=True)
                X = df_selected2.drop(['Latitude','Longitude'], axis=1)
                y_latitude = df_selected2['Latitude']
                y_longitude = df_selected2['Longitude']

                #Latitude
                X_train, X_test, y_train_latitude, y_test_latitude = train_test_split(
                            X, y_latitude, test_size=0.2, random_state=42
                        )
                scaler_latitude = MinMaxScaler()
                X_train_scaled = scaler_latitude.fit_transform(X_train)
                X_test_scaled = scaler_latitude.transform(X_test)
                rf_model_latitude = RandomForestRegressor(random_state=42)
                rf_model_latitude.fit(X_train_scaled, y_train_latitude)
                y_pred_latitude = rf_model_latitude.predict(X_test_scaled)
                mse_latitude = mean_squared_error(y_test_latitude, y_pred_latitude)

                #Longitude
                X_train, X_test, y_train_longitude, y_test_longitude = train_test_split(
                            X, y_longitude, test_size=0.2, random_state=42
                        )
                scaler_longitude = MinMaxScaler()
                X_train_scaled = scaler_longitude.fit_transform(X_train)
                X_test_scaled = scaler_longitude.transform(X_test)
                rf_model_longitude = RandomForestRegressor(random_state=42)
                rf_model_longitude.fit(X_train_scaled, y_train_longitude)
                y_pred_longitude = rf_model_longitude.predict(X_test_scaled)
                mse_longitude = mean_squared_error(y_test_longitude, y_pred_longitude)

                #PREDICT - Latitude
                test_case = pd.get_dummies(test_case)
                test_case_scaled = scaler_latitude.transform(test_case)
                lat_prediction = rf_model_latitude.predict(test_case_scaled)

                #PREDICT - Longitude
                test_case = pd.get_dummies(test_case)
                test_case_scaled = scaler_longitude.transform(test_case)
                long_prediction = rf_model_longitude.predict(test_case_scaled)

                txt = "Latitude: "+ str(lat_prediction[0]) + " Longitude:"+ str(long_prediction[0])
                htmlstr1=f"""<p style='background-color:green;
                                                        color:white;
                                                        font-size:18px;
                                                        border-radius:3px;
                                                        line-height:60px;
                                                        padding-left:17px;
                                                        opacity:0.6'>
                                                        {txt}</style>
                                                        <br></p>""" 
                st.markdown(htmlstr1,unsafe_allow_html=True)
            except:
                txt = "Invalid Input!!!!"
                htmlstr1=f"""<p style='background-color:red;
                                                        color:white;
                                                        font-size:18px;
                                                        border-radius:3px;
                                                        line-height:60px;
                                                        padding-left:17px;
                                                        opacity:0.6'>
                                                        {txt}</style>
                                                        <br></p>""" 
                st.markdown(htmlstr1,unsafe_allow_html=True)

            
        












import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    """Function to clean social media usage data"""
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame({
    "sm_li": s['web1h'].apply(clean_sm),
    "income": np.where(s['income'] > 9, np.nan, s['income']),
    "education": np.where(s['educ2'] > 8, np.nan, s['educ2']),
    "parent": s['par'].apply(clean_sm),
    "married": np.where(s['marital'] == 1, 1, 0),  # married is 1
    "female": np.where(s['gender'] == 2, 1, 0),   # female is 2
    "age": np.where(s['age'] > 98, np.nan, s['age'])
}).dropna()

# Ensuring income and education are in ordered numeric
income_categories = [1, 2, 3, 4, 5, 6, 7, 8, 9]
education_categories = [1, 2, 3, 4, 5, 6, 7, 8]

ss['income'] = pd.Categorical(ss['income'], categories=income_categories, ordered=True)
ss['education'] = pd.Categorical(ss['education'], categories=education_categories, ordered=True)
 
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=123) # set for reproducibility

log_reg = LogisticRegression(class_weight='balanced')

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_probabilities = log_reg.predict_proba(X_test)

# Streamlit

st.markdown("<h3 style='color: red;'>Are you on LinkedIn?</h3>", unsafe_allow_html=True)
st.markdown("##### Welcome to my predictive analysis app!")
st.markdown("My goal is to determine the likelihood that someone is a LinkedIn user based on various demographic factors.")
st.markdown("Please take a moment to fill out the following questions. Once you submit your responses, the app will provide you with a prediction of whether you are likely to be a LinkedIn user.")
st.markdown("<h6 style='color: red;'> Let's get started!</h6>", unsafe_allow_html=True)

#Income

income_household = {
    "Less than $10k": 1,
    "$10k to under $20k": 2,
    "$20k to under $30k": 3,
    "$30k to under $40k": 4,
    "$40k to under $50k": 5,
    "$50k to under $75k": 6,
    "$75k to under $100k": 7,
    "$100k to under $150k": 8,
    "$150k or more": 9
}

income_key =st.selectbox("What is your annual household income?",list(income_household.keys()))
income_value = income_household[income_key]

# Education Level

education_level = {
    "Less than high school (Grades 1-8 or no formal schooling)": 1,
    "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)": 2,
    "High school graduate (Grade 12 with diploma or GED certificate)": 3,
    "Some college, no degree (includes some community college)": 4,
    "Two-year associate degree from a college or university": 5,
    "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)": 6,
    "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)": 7,
    "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)": 8
}
 
education_key = st.selectbox("What is the highest level of education you have completed?", list(education_level.keys()))
education_value = education_level[education_key]

# Parent

parent_options = {
    "Yes": 1,
    "No": 0
}

parent_key = st.radio("Are you the parent of a child under 18 living in your home?", list(parent_options.keys()))
parent_value = parent_options[parent_key]
 
# Marital Status

marital_status_options = {
    "Married": 1,
    "Not Married": 0
}

marital_key = st.radio("Are you currently married?", list(marital_status_options.keys()))
marital_value = marital_status_options[marital_key]


# Gender

gender_options = {
    "Male": 0,
    "Female": 1
}
 
gender_key = st.radio("What is your gender?", list(gender_options.keys()))
gender_value = gender_options[gender_key]
 
# Age

age_value = st.number_input("How old are you?", min_value = 0, max_value = 98, step = 1)

# Input by user

input_by_user = pd.DataFrame({
    "income": [income_value],
    "education": [education_value],
    "parent": [parent_value],
    "married": [marital_value],
    "female": [gender_value],
    "age": [age_value]
})
 
 # Predictions
user_result = None
user_probability = None

if st.button('Predict'):
    user_result = log_reg.predict(input_by_user)[0]
    user_probability = log_reg.predict_proba(input_by_user)[0][1]
 
 # Display results

if user_result is not None:
    if user_result == 1:
        st.markdown("<h3 style='color: red;'>Yes, you are probably a LinkedIn user!</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: red;'>No, you are probably not a LinkedIn user.</h3>", unsafe_allow_html=True)
        
st.markdown(f"#### Your probability of using LinkedIn is {user_probability}")





st.markdown("Developed by Alicia Guzman, Georgetown University.")
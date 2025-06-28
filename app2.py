import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =============================
# Load data dan model
# =============================
df = pd.read_csv("E-commerce Customer Behavior.csv")
model = joblib.load("satisfaction_model.pkl")
encoders = joblib.load("encoder.pkl")
le_target = joblib.load("target_encoder.pkl")

# =============================
# Sidebar Menu Utama
# =============================
main_menu = st.sidebar.selectbox("Main Menu", ["Visualization Dashboard", "Customer Satisfaction Prediction"])

if main_menu == "Visualization Dashboard":
    st.title("📊 E-commerce Customer Behavior Visualization Dashboard")

    # Pilih visualisasi
    viz_menu = st.sidebar.radio("Select Visualization", [
        "Gender Distribution",
        "Average Total Spend",
        "Age Distribution",
        "Membership Distribution",
        "Satisfaction Level Distribution"
    ])

    # Filter City
    selected_cities = st.sidebar.multiselect(
        "Select City",
        sorted(df['City'].dropna().unique()),
        default=list(df['City'].dropna().unique())
    )
    
    # Filter Membership Type
    selected_membership = st.sidebar.multiselect(
        "Select Membership Type",
        sorted(df['Membership Type'].dropna().unique()),
        default=list(df['Membership Type'].dropna().unique())
    )

    # Filter Average Rating
    rating_min, rating_max = st.sidebar.slider(
        "Average Rating Range",
        0.0, 5.0,
        (float(df['Average Rating'].min()), float(df['Average Rating'].max())),
        step=0.1
    )

    # Filter Age
    age_min, age_max = st.sidebar.slider(
        "Age Range", 0, 100,
        (int(df['Age'].min()), int(df['Age'].max()))
    )

    # Terapkan filter
    filtered_df = df[
        (df['City'].isin(selected_cities)) &
        (df['Membership Type'].isin(selected_membership)) &
        (df['Average Rating'] >= rating_min) & (df['Average Rating'] <= rating_max) &
        (df['Age'] >= age_min) & (df['Age'] <= age_max)
    ]

    st.write(f"Showing data for City(s): **{', '.join(selected_cities)}**, "
             f"Membership Type(s): **{', '.join(selected_membership)}**, "
             f"Rating range: **{rating_min:.1f} - {rating_max:.1f}**, Age range: **{age_min} - {age_max}**")

    st.dataframe(filtered_df)

    # Visualisasi sesuai pilihan
    if viz_menu == "Gender Distribution":
        gender_count = filtered_df['Gender'].value_counts().reset_index()
        gender_count.columns = ['Gender', 'Count']
        fig = px.pie(gender_count, names='Gender', values='Count', title='Gender Distribution')
        st.plotly_chart(fig)

    elif viz_menu == "Average Total Spend":
        avg_spend = filtered_df.groupby('Membership Type')['Total Spend'].mean().reset_index()
        fig = px.bar(avg_spend, x='Membership Type', y='Total Spend', title='Average Total Spend per Membership Type')
        st.plotly_chart(fig)

    elif viz_menu == "Age Distribution":
        age_dist = filtered_df.groupby('Age').size().reset_index(name='Count')
        fig = px.line(age_dist, x='Age', y='Count', title='Customer Age Distribution')
        st.plotly_chart(fig)

    elif viz_menu == "Membership Distribution":
        membership_count = filtered_df['Membership Type'].value_counts().reset_index()
        membership_count.columns = ['Membership Type', 'Count']
        fig = px.bar(membership_count, x='Membership Type', y='Count', title='Membership Type Distribution')
        st.plotly_chart(fig)

    elif viz_menu == "Satisfaction Level Distribution":
        satisfaction_count = filtered_df['Satisfaction Level'].value_counts().reset_index()
        satisfaction_count.columns = ['Satisfaction Level', 'Count']
        fig = px.pie(satisfaction_count, names='Satisfaction Level', values='Count', title='Satisfaction Level Distribution')
        st.plotly_chart(fig)

elif main_menu == "Customer Satisfaction Prediction":
    st.title("🎯 Customer Satisfaction Prediction")

    st.markdown("Please enter the customer data below:")

    with st.form("input_form"):
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.slider("Age", 0, 100, 0)
        city = st.selectbox("City", sorted(df['City'].dropna().unique()))
        membership = st.selectbox("Membership Type", sorted(df['Membership Type'].dropna().unique()))
        total_spend = st.number_input("Total Spend", min_value=0.0, value=0.0)
        items_purchased = st.number_input("Items Purchased", min_value=0, value=0)
        average_rating = st.slider("Average Rating", 0.0, 5.0, 0.0)
        discount_applied = st.selectbox("Discount Applied", ['Yes', 'No'])
        days_since_last = st.slider(
            "Days Since Last Purchase",
            0,
            int(df['Days Since Last Purchase'].max()),
            0
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            'Gender': [gender],
            'Age': [age],
            'City': [city],
            'Membership Type': [membership],
            'Total Spend': [total_spend],
            'Items Purchased': [items_purchased],
            'Average Rating': [average_rating],
            'Discount Applied': [discount_applied],
            'Days Since Last Purchase': [days_since_last]
        }
        df_input = pd.DataFrame(input_data)

        for col in ['Gender', 'City', 'Membership Type', 'Discount Applied']:
            le = encoders[col]
            unseen = set(df_input[col]) - set(le.classes_)
            if unseen:
                le.classes_ = np.concatenate([le.classes_, list(unseen)])
            df_input[col] = le.transform(df_input[col])

        pred = model.predict(df_input)[0]
        label = le_target.inverse_transform([pred])[0]

        st.success(f"✅ Predicted Customer Satisfaction Level: **{label}**")


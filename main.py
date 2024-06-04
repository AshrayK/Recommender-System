# streamlit run main.py --server.enableXsrfProtection false

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm



# Define a function to load embeddings and filenames
@st.cache_data
def load_embeddings():
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
    return feature_list, filenames

# Load precomputed embeddings and filenames
feature_list, filenames = load_embeddings()

# Define and load the model
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tf.keras.Sequential([model, GlobalMaxPooling2D()])
    return model

model = load_model()

# Define a function to save uploaded files
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

@st.cache_data
def feature_extraction(img_path, _model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = _model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# Define a function to get recommendations based on extracted features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Placeholder for user data in session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'favorites': [],
        'search_history': [],
    }

# Define hardcoded user credentials
USER_CREDENTIALS = {
    "user1": "password1",
    "user2": "password2",
    "admin": "adminpassword"
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if 'user_credentials' not in st.session_state:
    st.session_state.user_credentials = USER_CREDENTIALS.copy()

# Define login page
def login_page():
    st.title('Login')
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.session_state.user_credentials and st.session_state.user_credentials[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Logged in as {username}")
            st.experimental_rerun()  # Refresh the app to show the appropriate page
        else:
            st.error("Invalid username or password")

# Define registration page
def registration_page():
    st.title('Register')
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if new_username in st.session_state.user_credentials:
            st.error("Username already taken. Please choose another.")
        else:
            st.session_state.user_credentials[new_username] = new_password
            st.success("Registration successful! You can now log in.")
            st.experimental_rerun()  # Refresh the app to show the login page

# Define logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully")
    st.experimental_rerun()  # Refresh the app to show the appropriate page

def main_page():
    st.title('Recommender System')
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://images.unsplash.com/photo-1577452160082-b3034692f3dd?q=80&w=1362&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: 100%;
            background-position: top left;
            background-repeat: no-repeat;
            background-attachment: local;
        }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            display_image = Image.open(uploaded_file)
            resized_image = display_image.resize((224, 224))
            st.image(resized_image)
            st.header(' Five suggestions for you to consider :')
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)  # Passed `model` as `_model`
            indices = recommend(features, feature_list)

            # Display recommendations
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(filenames[indices[0][0]])
            with col2:
                st.image(filenames[indices[0][1]])
            with col3:
                st.image(filenames[indices[0][2]])
            with col4:
                st.image(filenames[indices[0][3]])
            with col5:
                st.image(filenames[indices[0][4]])

            # Add to search history
            st.session_state.user_data['search_history'].append(uploaded_file.name)

            # Feedback mechanism
            st.subheader('Feedback')
            feedback = st.radio('Do you like these recommendations?', ('Select an option', 'Yes', 'No'))
            if feedback in ('Yes', 'No') and feedback != 'Select an option':
                st.success('Thank you for your feedback!')

            # Save favorite
            if st.button('Save to Favorites'):
                st.session_state.user_data['favorites'].append(uploaded_file.name)
                st.success('Saved to favorites!')
        else:
            st.header("Some error occurred while uploading the file.")



# Define the profile page layout and logic
def profile_page():
    st.title('Profile')
    st.subheader('Favorite Images')
    if st.session_state.user_data['favorites']:
        for favorite in st.session_state.user_data['favorites']:
            st.image(os.path.join("uploads", favorite))
    else:
        st.write("You have no favorite images yet.")

    st.subheader('Search History')
    if st.session_state.user_data['search_history']:
        for history in st.session_state.user_data['search_history']:
            st.write(history)
    else:
        st.write("You have no search history yet.")

# Define the admin page layout and logic
def admin_page():
    st.title('Admin')
    st.subheader('User Data')
    for user, password in st.session_state.user_credentials.items():
        st.write(f"Username: {user}, Password: {password}")

# Define the about page layout and logic
def about_page():
    st.title('About')
    st.markdown("""
        <style>
            .about-section {
                font-family: 'Arial', sans-serif;
                color: #FFFFFF;
                background-color: #333333;
                line-height: 1.6;
                padding: 20px;
                border-radius: 10px;
            }
            .about-section h2, .about-section h3 {
                margin-bottom: 20px;
                color: #FF5733;
            }
            .about-section h2 {
                font-size: 32px;
                margin-top: 40px;
            }
            .about-section h3 {
                font-size: 24px;
            }
            .about-section p {
                font-size: 18px;
                margin-bottom: 20px;
            }
            .team-section {
                margin-top: 50px;
            }
        </style>
        <div class="about-section">
            <h2>Welcome to Our Recommender System</h2>
            <p>
                Our recommender system is built using Streamlit and TensorFlow/Keras. We leverage advanced Machine Learning techniques to provide personalized recommendations, utilizing the power of nearest neighbor algorithms to bring you the best matches based on your preferences.
            </p>
            <p>
                Whether you are looking for fashion items, products, or other recommendations, our system is designed to help you find exactly what you are looking for with ease and accuracy.
            </p>
            <div class="team-section">
                <h2>Project Team</h2>
                <h3>Ashray Khosin</h3>
                <p>Registration Number: 12001742</p>
                <h3>Ujjwal Shukla</h3>
                <p>Registration Number: 12013254</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Define the home page layout and logic
def home_page():
    st.title('Home')
    st.markdown("""
        <style>
            .home-section {
                font-family: 'Arial', sans-serif;
                color: #FFFFFF;
                line-height: 1.6;
                background-color: #333333;
                padding: 20px;
                border-radius: 10px;
            }
            .home-section h3 {
                font-size: 28px;
                margin-bottom: 20px;
                color: #FF5733; /* Light orange color for headers */
            }
            .home-section p {
                font-size: 18px;
                margin-bottom: 20px;
                color: #CCCCCC; /* Light gray color for paragraphs */
            }
        </style>
        <div class="home-section">
            <h3>Welcome to the Fashion Recommendation System!</h3>
            <p>
                Our recommender system helps you find the perfect outfit based on your preferences. 
                Upload an image of a fashion item you like, and our system will suggest similar items from our collection.
            </p>
            <p>
                Utilizing advanced machine learning techniques and the ResNet50 model, our system extracts features from your uploaded image and finds the closest matches in our database.
            </p>
            <p>
                Start exploring now and discover the best fashion recommendations tailored just for you!
            </p>
        </div>
    """, unsafe_allow_html=True)

# Define the contact page layout and logic
def contact_page():
    st.title('Contact Us')
    st.markdown("""
        <style>
            .contact-section {
                font-family: 'Arial', sans-serif;
                color: #FFFFFF;
                line-height: 1.6;
                background-color: #333333;
                padding: 20px;
                border-radius: 10px;
            }
            .contact-section h3 {
                font-size: 28px;
                margin-bottom: 20px;
                color: #FFFFFF; /* Blue color for headers */
            }
            .contact-section p {
                font-size: 18px;
                margin-bottom: 20px;
            }
            .contact-section ul {
                list-style-type: none;
                padding: 0;
            }
            .contact-section ul li {
                margin-bottom: 10px;
            }
            .contact-section strong {
                color: #FF5733; /* Light orange color for strong elements */
            }
        </style>
        <div class="contact-section">
            <h3>Get in Touch</h3>
            <p>If you have any questions or need further information, feel free to reach out to us:</p>
            <ul>
                <li>Email: <a href="mailto:ashraykhosin@gmail.com">ashraykhosin@gmail.com</a></li>
                <li>Email: <a href="mailto:ujjwal7017@gmail.com">ujjwal7017@gmail.com</a></li>
                <li>Contact Number: <strong>6396627009</strong></li>
                <li>Postal Address: 642, Braham Nagar, Auraiya, Uttar Pradesh, 206122</li>
            </ul>
            <p>We'll do our best to respond to your inquiries as quickly as possible.</p>
        </div>
    """, unsafe_allow_html=True)

# Define the help and support page layout and logic
def help_page():
    st.title('Help & Support')
    st.markdown("""
        Help & Support
        If you need assistance or have any questions, you're in the right place. Below you'll find answers to frequently asked questions, user guides, and contact information for further support.

        <h4>Frequently Asked Questions</h4>
        <p><strong>Q: How do I upload an image?</strong></p>
        <p>A: To upload an image, go to the 'Main' page, click on the 'Upload an image' button, and select the image file from your computer.</p>

        <p><strong>Q: What image formats are supported?</strong></p>
        <p>A: Our system supports JPEG, PNG, and BMP formats. Make sure your image is in one of these formats for a successful upload.</p>

        <p><strong>Q: How do I save an image to my favorites?</strong></p>
        <p>A: After uploading an image and viewing the recommendations, you can save an image to your favorites by clicking the 'Save to Favorites' button.</p>

        <p><strong>Q: How can I provide feedback on the recommendations?</strong></p>
        <p>A: After viewing the recommendations, you can provide feedback by selecting 'Yes' or 'No' under the 'Feedback' section.</p>

        <p><strong>Q: What should I do if I encounter an error?</strong></p>
        <p>A: If you encounter an error, try reloading the page and attempting the action again. If the problem persists, please contact our support team.</p>

        <h4>User Guides</h4>
        <p><strong>Uploading an Image:</strong></p>
        <ul>
            <li>Navigate to the 'Main' page.</li>
            <li>Click on the 'Upload an image' button.</li>
            <li>Select the image you want to upload from your device.</li>
            <li>Wait for the image to be processed and view the recommendations.</li>
        </ul>

        <p><strong>Saving to Favorites:</strong></p>
        <ul>
            <li>After uploading an image and receiving recommendations, click the 'Save to Favorites' button below the image.</li>
            <li>The image will be added to your favorites list, which you can view anytime.</li>
        </ul>

        <p><strong>Providing Feedback:</strong></p>
        <ul>
            <li>After viewing the recommendations, select 'Yes' or 'No' under the 'Feedback' section to indicate if you liked the recommendations.</li>
            <li>Your feedback helps us improve the recommendation system.</li>
        </ul>

        <h4>Contact Support</h4>
        <p>If you need further assistance or have any questions, please feel free to reach out to us:</p>
        <ul>
            <li>Ashray Khosin: <a href="mailto:ashraykhosin@gmail.com">ashraykhosin@gmail.com</a></li>
            <li>Ujjwal Shukla: <a href="mailto:ujjwal7017@gmail.com">ujjwal7017@gmail.com</a></li>
            <li>Contact Number: <strong>6396627009</strong></li>
        </ul>
        <p>We'll do our best to respond to your inquiries as quickly as possible.</p>
    """, unsafe_allow_html=True)

# Add a sidebar for navigation
if st.session_state.logged_in:
    pages = ["Home", "Main", "Profile", "About", "Contact", "Help & Support", "Logout"]
    if st.session_state.username == "admin":
        pages.append("Admin")
else:
    pages = ["Login", "Register"]

page = st.sidebar.selectbox("Choose a page", pages)

# Display the selected page
if st.session_state.logged_in:
    if page == "Home":
        home_page()
    elif page == "Main":
        main_page()
    elif page == "Profile":
        profile_page()
    elif page == "About":
        about_page()
    elif page == "Contact":
        contact_page()
    elif page == "Help & Support":
        help_page()
    elif page == "Logout":
        logout()
    elif page == "Admin":
        admin_page()
else:
    if page == "Login":
        login_page()
    elif page == "Register":
        registration_page()
    else:
        st.warning("Please login or register to access other pages.")

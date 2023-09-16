
import streamlit as st
import nltk
nltk.download('punkt')
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
# import cv2
# import numpy as np


# # Create a circular mask using NumPy
# def create_circle_mask(dim):
#     mask = np.zeros((dim, dim), np.uint8)
#     cv2.circle(mask, (dim // 2, dim // 2), dim // 2, 255, -1)
#     return mask

# mask = create_circle_mask(400)

# Load the emotion lexicon (Assuming you have a file named dataemo.csv)
emotion_lexicon = pd.read_csv("dataemo.csv")
emotion_lexicon.fillna(0, inplace=True)
emotion_lexicon.drop(['Emotion', 'Number of Records'], axis=1, inplace=True)

# Convert specified columns to binary
cols_to_convert = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'Negative', 'Positive']
for col in cols_to_convert:
    emotion_lexicon[col] = emotion_lexicon[col].apply(lambda x: 1 if x != 0 else 0)

# Streamlit UI
st.title("Emotion Analyzer")

# Text input
text_input = st.text_area("Enter your text here:")

# File uploader
uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")
    dominant_emotion_words = [] 
# Zoom slider
# zoom_level = st.slider('Zoom Level:', min_value=0.5, max_value=2.0, value=1.0, step=0.1)

if text_input:

    # Tokenize and count emotions in the text
    words = word_tokenize(text_input.lower())
    emotion_count = {col: 0 for col in emotion_lexicon.columns if col != 'Word'}
    dominant_emotion_words = [] 
    for word in words:
        if word in emotion_lexicon['Word'].values:
            for emotion in emotion_lexicon.columns:
                if emotion != 'Word':
                    emotion_count[emotion] += emotion_lexicon.loc[emotion_lexicon['Word'] == word, emotion].values[0]
                    
    # Display dominant emotion and plots


    # Determine the dominant emotion for the text
    dominant_emotion = max(emotion_count, key=emotion_count.get)
    # Collect words contributing to the dominant emotion
    for word in words:
        if word in emotion_lexicon['Word'].values:
            for emotion in emotion_lexicon.columns:
                if emotion != 'Word':
                    if emotion_lexicon.loc[emotion_lexicon['Word'] == word, emotion].values[0] > 0:
                        if emotion == dominant_emotion:
                            dominant_emotion_words.append(word)
    st.markdown(
    """
    <style>
        .guide {
            font-family: 'Times New Roman', Times, serif;
        }
    </style>
    <div class="guide">

    ### How The Emotion Analyzer Works

    This Emotion Analyzer is designed to process textual input and identify the most dominant emotion exhibited in the text. Below is an overview of how the algorithm and code work:

    1. **Import Required Libraries**: Libraries such as Pandas, NLTK, and Matplotlib are imported for data manipulation, text processing, and plotting.

    2. **Load Emotional Lexicon**: The lexicon (usually in a CSV file) containing words associated with different emotions is loaded. This lexicon is the main source for the emotion analysis. 

        - **Source of Emotion Lexicon Dataset**: The lexicon usually comes from a reputable source, such as the NRC Emotion Lexicon or other academic publications that have tabulated words against various emotions.

    3. **Text Input**: The application accepts text input from the user either directly through a text box or by uploading a text file.

    4. **Tokenization**: The input text is tokenized into individual words.

    5. **Emotion Count**: Each tokenized word is checked against the emotional lexicon to identify and count the associated emotions.

    6. **Determine Dominant Emotion**: The emotion with the highest count is determined to be the dominant emotion of the text.

    7. **Visualization**: A bar graph is plotted to show the frequency of each emotion in the text, and a word cloud is generated to visualize the most frequently occurring words.

    </div>
    """, 
    unsafe_allow_html=True)
    st.markdown(
    """
    <style>
        .funny-text {
            font-size: 40px;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            color: #FF69B4;
            text-shadow: 2px 2px #FFD700;
        }
    </style>
    <div class="funny-text">Silly it is not A.I 😂</div>
    """, 
    unsafe_allow_html=True)

    st.header(f"The dominant emotion in the text is {dominant_emotion}.")
    # Show the top 5 words contributing to the dominant emotion in a table
    word_freq = pd.DataFrame(dominant_emotion_words, columns=["Words"])
    word_freq = word_freq["Words"].value_counts().reset_index()
    word_freq.columns = ["Words", "Frequency"]
    word_freq = word_freq.head(5)  # Top 5 words
    st.write(f"Top 5 words contributing to {dominant_emotion}:")
    st.table(word_freq)
   # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_input)

    st.subheader("Word Cloud")
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig)
    # wordcloud = WordCloud(width=800, height=400, background_color='white', mask=mask).generate(text_input)

    # st.subheader("Word Cloud")
    # fig, ax = plt.subplots(figsize=(8 * zoom_level, 4 * zoom_level))  # Change figure size according to zoom level
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis('off')
    # st.pyplot(fig)



    st.subheader("Emotion Distribution")
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
    ax.bar(emotion_count.keys(), emotion_count.values(), color='dodgerblue')

    # Annotate the bars with the actual values
    for i, v in enumerate(emotion_count.values()):
        ax.text(i, v + 0.1, str(v), ha='center')

    plt.xlabel("Emotions", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-labels for readability
    plt.title("Frequency of Emotions in Text", fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid for better visualization
    plt.tight_layout()  # Adjust layout for better fit

    st.pyplot(fig)


    st.markdown(
                "<h2 style='font-family:Times New Roman, Times, serif; font-style:italic;'>Data Scientist behind the app</h2>",
                unsafe_allow_html=True,
            )

    st.markdown(
                "<div style='font-family:Times New Roman, Times, serif; font-style:italic;'>"
                "Hello everyone, this app is created and managed by Sajid Hafeez, Data scientist at Rprogrammers.com.<br>"
                "I offer services related to Data science and statistical analysis using R, Python, Stata, SPSS, Weka and Power BI. Feel free to contact me on the following."
                "</div>",
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)

    email_logo_base64 = image_to_base64("email.png")
    linkedin_logo_base64 = image_to_base64("whats.png")
    website_logo_base64 = image_to_base64("web.png")

    col1.markdown(
                f"""
                <div style='font-family:Times New Roman, Times, serif; font-style:italic;'>
                    <p><img src='data:image/png;base64,{email_logo_base64}' style='width:20px; vertical-align:middle;'/> Email: <a href='mailto:Sajidhafeex@gmail.com'>Sajidhafeex@gmail.com</a></p>
                    <p><img src='data:image/png;base64,{linkedin_logo_base64}' style='width:20px; vertical-align:middle;'/> LinkedIn: <a href='https://www.linkedin.com/in/sajid-hafeex'>https://www.linkedin.com/in/sajid-hafeex</a></p>
                    <p><img src='data:image/png;base64,{website_logo_base64}' style='width:20px; vertical-align:middle;'/> Website: <a href='https://Rprogrammers.com'>https://Rprogrammers.com</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    col2.image("dpp.png")

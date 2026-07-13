# ==========================================================
# One-to-Many RNN
# Text Generation using Simple RNN
# Streamlit + TensorFlow
# ==========================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SimpleRNN,
    Dense,
    RepeatVector,
    TimeDistributed
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==========================================================
# Files
# ==========================================================

MODEL = "one_to_many_rnn.keras"
TOKENIZER = "tokenizer.pkl"
DATASET = "technology_sentences.csv"

MAX_WORDS = 5000

# ==========================================================
# Streamlit Page
# ==========================================================

st.set_page_config(
    page_title="One-to-Many RNN",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 One-to-Many RNN")
st.write("### Text Generation using Encoder-Decoder Simple RNN")

# ==========================================================
# Load Dataset
# ==========================================================

if not os.path.exists(DATASET):
    st.error("technology_sentences.csv not found.")
    st.stop()

df = pd.read_csv(DATASET)

keywords = df["keyword"].astype(str).tolist()
sentences = df["sentence"].astype(str).tolist()

# ==========================================================
# Tokenizer
# ==========================================================

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    keywords + sentences
)

vocab_size = len(tokenizer.word_index) + 1

# ==========================================================
# Input Sequences
# ==========================================================

X = tokenizer.texts_to_sequences(keywords)

max_input_len = max(len(i) for i in X)

X = pad_sequences(
    X,
    maxlen=max_input_len,
    padding="post"
)

# ==========================================================
# Output Sequences
# ==========================================================

Y = tokenizer.texts_to_sequences(sentences)

max_output_len = max(len(i) for i in Y)

Y = pad_sequences(
    Y,
    maxlen=max_output_len,
    padding="post"
)

Y = to_categorical(
    Y,
    num_classes=vocab_size
)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Training Function
# ==========================================================

def train_model():

    st.info("Training model... Please wait.")

    encoder_inputs = Input(
        shape=(max_input_len,)
    )

    x = Embedding(
        input_dim=vocab_size,
        output_dim=128
    )(encoder_inputs)

    encoded = SimpleRNN(
        128
    )(x)

    repeated = RepeatVector(
        max_output_len
    )(encoded)

    decoded = SimpleRNN(
        128,
        return_sequences=True
    )(repeated)

    outputs = TimeDistributed(
        Dense(
            vocab_size,
            activation="softmax"
        )
    )(decoded)

    model = Model(
        encoder_inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=4,
        validation_data=(X_test, y_test),
        verbose=1
    )
        # -----------------------------------------
    # Save Model
    # -----------------------------------------

    model.save(MODEL)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    st.success(f"Model Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# Train Only Once
# ==========================================================

if not os.path.exists(MODEL):
    train_model()

# ==========================================================
# Cached Model Loader
# ==========================================================

@st.cache_resource
def get_model():

    return load_model(
        MODEL,
        compile=False
    )

# ==========================================================
# Cached Tokenizer Loader
# ==========================================================

@st.cache_resource
def get_tokenizer():

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer

# ==========================================================
# Generate Sentence
# ==========================================================

def generate_sentence(keyword):

    model = get_model()

    tokenizer = get_tokenizer()

    seq = tokenizer.texts_to_sequences(
        [keyword.lower()]
    )

    seq = pad_sequences(
        seq,
        maxlen=max_input_len,
        padding="post"
    )

    prediction = model.predict(
        seq,
        verbose=0
    )

    prediction = np.argmax(
        prediction,
        axis=-1
    )[0]

    reverse_word_index = {
        value: key
        for key, value in tokenizer.word_index.items()
    }

    sentence = []

    for index in prediction:

        if index == 0:
            continue

        word = reverse_word_index.get(index)

        if word is None:
            continue

        if word == "<OOV>":
            continue

        sentence.append(word)

    result = " ".join(sentence)

    return result
# ==========================================================
# Streamlit User Interface
# ==========================================================

st.markdown("---")

keyword = st.text_input(
    "Enter Keyword",
    value="Python"
)

if st.button("Generate Sentence"):

    if keyword.strip() == "":

        st.warning("Please enter a keyword.")

    else:

        try:

            sentence = generate_sentence(keyword)

            st.subheader("Generated Sentence")

            if sentence.strip() == "":

                st.error(
                    "Unable to generate a sentence for the given keyword."
                )

            else:

                st.success(sentence)

        except FileNotFoundError:

            st.error(
                "Model or tokenizer file not found. Please train the model first."
            )

        except Exception as e:

            st.error(
                "Error while generating sentence."
            )

            st.exception(e)

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "One-to-Many Recurrent Neural Network (RNN) | "
    "Text Generation using Encoder-Decoder Simple RNN"
)
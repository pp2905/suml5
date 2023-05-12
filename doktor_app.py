# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model2.sv"
model = pickle.load(open(filename, "rb"))
# otwieramy wcześniej wytrenowany model

objawy = {1: "Wymioty", 2: "Sraczka", 3: "Ból głowy", 4: "Ból nogi", 5: "Ból dupy"}
leki_radio = {1: "Na sen", 2: "Na głowę", 3: "przeciw nikotynie", 4: "na uspokojenie"}


def main():

    st.set_page_config(page_title="App Titanic")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://i1.kwejk.pl/k/obrazki/2017/12/bad1a96d3e90c2f93be2e0c2e67f1878.jpg")

    with overview:
        st.title("App Choroba")

    with left:
        sex_radio = st.radio("Leki", list(leki_radio.keys()), format_func=lambda x: leki_radio[x])
        pclass_radio = st.radio("Objawy", list(objawy.keys()), index=2, format_func=lambda x: objawy[x], )

    with right:
        wiek_slider = st.slider("Wiek", value=1, min_value=1, max_value=90)
        wzrost_slider = st.slider(
            "Wzrost", min_value=150, max_value=215
        )
        choroby = st.slider(
            "ile chiorób współistniejących", min_value=0, max_value=5
        )

    data = [
        [
            pclass_radio,
            sex_radio,
            wiek_slider,
            wzrost_slider,
            choroby,
        ]
    ]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba zachoruje na przewlekłe zapalenie płata czołowego nadgarstka")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write(
            "Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100)
        )


if __name__ == "__main__":
    main()

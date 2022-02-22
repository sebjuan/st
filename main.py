
import streamlit as st
from ml import bert_model, df, get_random_plot


user_input =''
max_result = st.slider('max_result', 1, 50, 3)

st.markdown("Go on with your imaginary plot... like : ")
st.markdown(f"""
* "a banker wants to change his life"
* "a terrorist tries to kidnap the son of the queen"
* ...
#
You can even just use it as a search engine, try : 
* "submarine"
* "dinosaur"
#
...or pick a random synopsis of our database
""")


plot = None
if st.button("I can't think of anything, give me a (another) random plot"):
    plot = get_random_plot()

user_input = st.text_area("",plot if plot else '')

for i in bert_model.predict(user_input, max_result):
    if not user_input:
        st.write()
    else:
        st.header(df.loc[i,'TITLE'])
        s = df.loc[i,'RAW_PLOT']
        st.write(df.loc[i,'RAW_PLOT'])









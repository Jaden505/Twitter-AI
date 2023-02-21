import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import data


st.set_option('deprecation.showPyplotGlobalUse', False)

class Dashboard:
    def __init__(self):
        d = data.Data()
        self.df = d.get_local_data()
        self.selected_user = None
        self.MAX_USERS = 50

        self.display_text()

    def generate_wordcloud(self):
        if self.selected_user is None:
            return

        text = ' '.join(self.df[self.df['user'] == self.selected_user]['text'].values)
        wordcloud = WordCloud(background_color="white", max_words=2000, width=800, height=400).generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()

    def display_text(self):
        st.title('Wordcloud')
        st.write('Generate a wordcloud from the tweets of a selected user.')

    def select_user(self):
        self.selected_user = st.selectbox('Select a user', self.df['user'].unique()[:self.MAX_USERS], key='user')

if __name__ == '__main__':
    dash = Dashboard()
    dash.select_user()
    dash.generate_wordcloud()
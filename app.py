"""

The main goal of this project is to create a streamlit app which which uses a recommendation
model to recommend books to the user along with the book which user has searched for based of other users
rating on certain books.

The flow of the app.py is as follows : 
1. Importing the required libraries.
2. Unpickling the pickled files.
3. Model fitting.
4. Creating recommeder function.
5. Implementing model into web page.

"""

#--------------Importing the required libraries--------------

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import wget

#--------------Unpickling the pickled files-----------------

rating_table = pickle.load(open("pickled/rating_table.pkl","rb"))
books_image_data = pickle.load(open("pickled/books_image_data.pkl","rb"))

#----------------------Model fitting------------------------

sparse_matrix = csr_matrix(rating_table)
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_matrix)

#-------------------Creating recommeder function---------------

#This function takes a book name >><str> and returns a list of 5 books and their image link.
def recommend(book_name):
  """
  @param -- <str> book_name.
  @returns -- Two <list> of <str> 
  """
  recommended_books = []
  image_url = []
  #Extract the index of input book.
  book_index = np.where(rating_table.index==book_name)[0][0]
  distances , suggestions = model.kneighbors(rating_table.iloc[book_index,:].values.reshape(1,-1),n_neighbors=6)

  #Convert suggested 2d array into 1d array.
  suggestions = np.ravel(suggestions, order='C')

  #Get recommended books name.
  for i in suggestions:
    recommended_books.append(rating_table.index[i])

  #Get image link of those recommended books.
  for i in recommended_books:
    image_url.append(books_image_data[books_image_data["title"] == i ].image.to_string(index=False))
    
  return recommended_books,image_url
#---------------Implementing model into web page-----------------

#Refer streamlit documentation for frontend.

st.subheader("Collaborative Filtering Based Books Recommender Engine") #Title

#Extracting the books name from the loaded pickled rating table
books_name = rating_table.index.to_list()
#Dropdown select menu
selected_book = st.selectbox(
     'Search Your Book Here',
     books_name)
    


if st.button('Search'):
    books,images = recommend(selected_book) 
    #This image download step is improvised to bypass the problem of herokun not showing images through link.
    img1 = wget.download(images[0])
    img2 = wget.download(images[1])
    img3 = wget.download(images[2])
    img4 = wget.download(images[3])
    img5 = wget.download(images[4])
    img6 = wget.download(images[5])

    container1 =st.container()
    container1.subheader("You Searched For:")
    container1.markdown(books[0])
    container1.image(img1,width=120)

    st.subheader("Users Also Liked:")
    col1, col2, col3,col4,col5 = st.columns(5)

    with col1:
        st.text(books[1])
        st.image(img2,width=100)
    with col2:
        st.text(books[2])
        st.image(img3,width=100)
    with col3:
        st.text(books[3])
        st.image(img4,width=100)
    with col4:
        st.text(books[4])
        st.image(img5,width=100)
    with col5:
        st.text(books[5])
        st.image(img6,width=100)





import os
import re
from types import new_class
import pandas as pd
from pyrsistent import b
from stop_words import get_stop_words as words_to_remove


class GetBookData:
    def __init__(self):
        """
        
        Parameters:
        -----------
        path: str
            A path to where the book data is
    
        Returns:
        --------
        """

    def get_books(self, path:str):
        """
        This function will get all the files in a directory that have a txt extension. It will
        select them all and add them to a list as `./path/file.txt`. The list contaiting all the 
        paths will be returned

        Parameters:
        -----------
        path: str
            A path to where the book data is
    
        Returns:
        --------
        """
        books = []
        # loop through the directory
        for file in os.listdir(path):
            # select all txt files
            if file.endswith(".txt"):
                book = file
                books.append(f'{path}{book}')
        return books

    def open_book(self, book_path):
        with open(book_path, 'r') as file:        
            book = file.read().replace('\n', ' ')  
        return book



    def clean_book(self, book):
        cleaned = re.sub(r'[\.!#$%*?[()@,:/;"{}+=-]', ' ', book)
        clean_nums = re.sub(r'[0-9]', ' ', cleaned)
        tokens = clean_nums.split()
        tokens = [token.lower() for token in tokens]
        return ' '.join(tokens)

    def get_data(self, paths):
        data = {'book': [], 'words': []}
        for i in range(len(paths)):
            book_name = paths[i]
            book = self.open_book(book_name)
            clean_book = self.clean_book(book)
            data['book'].append(book_name)
            # print(book)
            data['words'].append(clean_book)
            # print(clean_book)
            # print()

        return data


    def write_data_to_csv(self, data):
        """
        This function will save our dataframe into a csv file.

        Parameters
        ----------
        data: dict
            A dictionary containing some data

        Returns
        -------
        None
        """
        # print(data)
        data_frame = pd.DataFrame(data=data)
        data_frame.to_csv('data.csv', index=False)
        return 0

book_class = GetBookData()
paths = book_class.get_books('./data/')
data = book_class.get_data(paths)
book_class.write_data_to_csv(data)

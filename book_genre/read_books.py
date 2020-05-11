import os
import re


def read_book(book):
    """
    This function takes in a book as an argument and will return a list
    of all the words in that book

    Parameters
    ----------
    book; str
        This is a string which contains the pathway and name of te book
        that we will open and read 

    Returns
    -------
    list
        Returns a list of all the words in the book
    """
    # open book
    with open(book, 'r') as file:        
        data = file.read().replace('\n', ' ')  
        # get book into list form with all lowercase words
        tokens = data.split()
        tokens = [token.lower() for token in tokens] 
        tokens_index = list(set(tokens))
    return tokens_index


def common_words():
    """
    This function accepts the pathway to some book data. The data Should be formated so that 
    the names start with a category name i.e `0_edgar_allen_poe.txt`. This funtction returns a dict
    where the key of the dict is the book name and the value is all words. 

    Parameters:
    -----------
    dict: str
        A pathway in the file system where the data is.


    Returns:
    --------
    Int
       This int will show many words they have in common 
    """
    return 0

def download_book(link:str):
    """
    This function takes in a book as an argument and will return the book without
    any puntctuation

    Parameters
    ----------
    link; str
        This is a string which will be a link to download a book

    Returns
    -------
    list
        Returns a list of all the words in the book
    """

    return book

def get_average_word_length(book):
    """
    This function accepts a book as its argument and will return the average length
    of every word in the book

    Parameters:
    -----------
    book: list
        A list of all the words in the book

    Returns:
    --------
    int
        This int will be the average length of a word
    """
    average_word = 0
    for i in range(len(book)):
        average_word += len(book[i])
    

    return (average_word/len(book))

def get_length(book):
    """
    This function takes in a book as its arugment and returns the length 
    of the book which is the total ammount of words in the book

    Parameters
    ----------
    book; list
        This is a list of all the words in the book 

    Returns
    -------
    int 
        This function returns a int which is the total number of words in 
        the book
    """
    return len(book)




def most(book:list):
    """
   This function takes in a book as its arugment which is a list of all the words 
   in the book and will return a list of the most occuring words in the book

    Parameters
    ----------
    book; list
        This is a list of all the words in the book 

    Returns
    -------
    list
        This function returns a list of the most occruring words in the book
    """
    most_occuring = book
    return most_occuring


def clean_book(book):
    """
    This function takes in a book as an argument and will return the book without
    any puntctuation

    Parameters
    ----------
    book; list
        This is a list of all the words in the book 

    Returns
    -------
    list
        This function returns a list of all the words in the book without 
        any punctuation
    """
    cleaned = re.sub(r'[\.!#$%*()@,:/;"{}+=-]', ' ', book)
    cleaned = re.sub(r'[0-9]', ' ', cleaned)
    

    return cleaned



def rmv_long_short(book):
    """
    This function takes in a book as its argument and will remove all the very long 
    words as well as all the very short words. The function will then return a list
    of all the words in the book expect the very long and very short ones. 

    Parameters
    ----------
    book; list
        This is a list of all the words in the book 

    Returns
    -------
    list
        This function will return a list of all the words in the book but will all the
        very short and very long words removed
    """
    some_list = []
    clean_book = []
    for i in range(len(book)):
        word = book[i]
        if word not in some_list:
            clean_book.append(word)

    return clean_book


def get_length_all(dict_of_books: dict):
    """
    This function takes in a dictionary of books and prints thet length of the books

    Parameters
    ----------
    dict_of_books; dict
        This is a dictionary of books where the key is title of the book and value is
        all words in the book

    Returns
    -------
    None
        This function returns nothing it just prints the name of the book and 
        number of words in it
    """
    length_dict = {}
    for book in dict_of_books:
        length_of_book = get_length(dict_of_books[book])
        length_dict[book] = length_of_book
        print(f"{book}, {length_of_book} ")
     
    return length_dict

def make_dict_of_book(pathway):
    """
    This function accepts the pathway to some book data. The data Should be formated so that 
    the names start with a category name i.e `0_edgar_allen_poe.txt`. This funtction returns a dict
    where the key of the dict is the book name and the value is all words. 

    Parameters:
    -----------
    pathway: str
        A pathway in the file system where the data is.


    Returns:
    --------
    dict
        The dict contains the title as a key and all its words as a value. 
    """
    book_dict = {}
    # loop through the directory
    for file in os.listdir(f"./{pathway}"):
        # select all txt files
        if file.endswith(".txt"):
            book_name = file
            # read_books() returns a list of words in book
            book_list = read_book(os.path.join(f"./{pathway}/" + book_name))
            # add the name of the book (without .txt) and its words
            book_dict[book_name[:-4]] = book_list


    return book_dict


def get_features():
    """
    This function is the driver function and takes in no arguments. The purpose of 
    this function is to get features from a book i.e `book length`etc. This function 
    takes in no arguments but will return all the features it has collected. First the 
    function calls the book create dictionary function to get all our data of books.
    Then we will do some cleaning and.

    Parameters:
    -----------
    None


    Returns:
    --------
    
        The dict contains the title as a key and all its words as a value. 
    """
    # dictionary with words and book
    book_dict = make_dict_of_book("data")
    # dictionary with book and length
    length_dict = get_length_all(book_dict)
    # average_word = get_average_word_length(book_dict)

    return length_dict, average_word




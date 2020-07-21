import os
import re
import pandas as pd
from stop_words import get_stop_words as words_to_remove



def make_dict_of_book(pathway:str) -> dict:
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
            book = read_book(os.path.join(f"./{pathway}/" + book_name))
            book_list = clean_book(book)
            # clean = rmv_stop_words(book_list)
            # add the name of the book (without .txt) and its words
            book_dict[book_name[:-4]] = book_list

    return book_dict



def read_book(book):
    with open(book, 'r') as file:        
        data = file.read().replace('\n', ' ')  
    return data



def clean_book(book:list) -> list:
    """
    This function takes in a list as its argument which contains the words to a book.
    The function will clean the list by removing some punctuation, numbers, and make 
    all words lowercase. The function will then return the cleaned version of the book

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
    cleaned = re.sub(r'[\.!#$%*?[()@,:/;"{}+=-]', ' ', book)
    clean_nums = re.sub(r'[0-9]', ' ', cleaned)
    tokens = clean_nums.split()
    tokens = [token.lower() for token in tokens]
    words = rmv_stop_words(tokens)
    return tokens



###### This function doesnt do anything
def rmv_stop_words(book:list) -> list:
    """
    This function takes in a list containing all the words to a book as its argument.
    The function will remove stop words ie `the`, `and`, `they` and will return a list
    without the stop words

    Parameters
    ----------
    book; list
        This is a list of all the words in the book 

    Returns
    -------
    list
        This function will return a list of all the words in the book without stop
        words
    """
    clean_book = []
    stop_words = words_to_remove()
    for i in range(len(stop_words)):
        if stop_words[i] not in book:
            clean_book.append(stop_words[i])
        else:
            pass

    # print(stop_words)
    return clean_book


def count_most_occuring_words(books_words:list) -> dict:
    """
    This function takes in a list of words from a book as its argument. 

    Parameters:
    -----------
    books_words: list
        A list of words belonging to a book


    Returns:
    --------
    dict
       This int will show many words they have in common 
    """
    word_counter = {}
    common = {}
    # count the frequency of the words
    for word in books_words:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1

    common_words = sorted(word_counter, key = word_counter.get, reverse = True)

    # get least and most common
    top_five = common_words[:5]
    bottom_five = common_words[-5:]
    # join lists together
    least_and_most_common = top_five + bottom_five

    # make a dictionary with least and most common words
    for i in range(len(least_and_most_common)):
        word = least_and_most_common[i]
        if word in word_counter:
            # add the word as a key and its frequency as its value
            common[word] = word_counter[word]
    
    return common



def most_common_words(book_dict:dict) -> dict:
    """
    This function takes in a dictionary as its argument in which the dictionary
    contains the title of a book as its key and the words as its value. The function
    passess the value to another function to then count the frequency of the words 
    for that book. Once we have the frequency of the words for each book we add it
    to a dictionary. We then return the dictionary witht the word frequency.

    Parameters:
    -----------
    dict: str
        A pathway in the file system where the data is.


    Returns:
    --------
    dict
       This int will show many words they have in common 
    """
    frequency_dict = {}
    for book in book_dict:
        book_word_list = book_dict[book]
        words = count_most_occuring_words(book_word_list)
        frequency_dict[book] = words

    return frequency_dict



###### This function doesnt do anything yet
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


def get_average_word_length(book) -> int:
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
    length_of_all = 0
    for i in range(len(book)):
        length_of_all += len(book[i])
    
    return int(length_of_all/len(book))


#new comment

def get_length(book:list) -> int:
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



def get_length_all(dict_of_books: dict) -> dict:
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
     
    return length_dict

###### This function doesnt do anything yet
def count_repeat(book_dict: dict):
    """
    This function takes in dictionary as its argument and checks if words in books are repeated.
    This function will return a dictionary with the avarage times a books words repeat.


    Parameters
    ----------
    dict_of_books; dict
        This is a dictionary of books where the key is title of the book and value is
        all words in the book

    Returns
    -------
    dict
        This function returns a dictionary with the name of the book as its key and the 
        average times the words repeate as its value.
    """
    repeat_dict = {}
    seen = []
    average_repeate = 0
    for book in book_dict:
        words = book_dict[book]
        for i in range(len(words)):
            if i in words:
                average_repeate += 1
                seen.append(i)
            else:
                seen.append(i)
        repeat_dict[book] = (average_repeate/len(words))
        average_repeate = 0
    
    return repeat_dict


def get_book_length_noreapeat(book_dict:dict) -> dict:
    """
    This function takes in a dictionary of books and prints thet length of the books

    Parameters
    ----------
    book_dict; dict
        This is a dictionary of books where the key is title of the book and value is
        all words in the book

    Returns
    -------
    dict
        This function returns a dictionary with the name as the key and legnth of book with
        no repeates as its value
    """
    length_norepeat = {}
    for book in book_dict:
        # call function that returns length of lists
        set_words = set(list(book_dict[book]))
        # add to dictionary
        length_norepeat[book] = len(set_words)

    return length_norepeat





def to_dataframe(length_dict:dict, book_dict:dict, no_repeat:dict, least_common:dict):
    """
    This functions gets passed in 3 dictionaries containing some data from
    the books which are the features. This function will then put that into 
    a pandas dataframe. 

    Parameters:
    -----------
    length_dict: dict
        This is a dictionary with the name of the book as its key and the total
        number of words as its value


    Returns:
    --------
    Pandas Dataframe
        
    """
    data = []
    books = []
    columns = []
    
    for book in length_dict:
        data.append(  [book, length_dict[book], get_average_word_length(book_dict[book]), no_repeat[book], book_dict[book], least_common[book] ] )



    for book in least_common:
        # print(least_common[book]) # nested dictionary
        # print(book) # name of book
        for word in least_common[book]:
            


    # words_df = pd.DataFrame(book, columns=columns)
        
    

    df = pd.DataFrame(data, columns = ['Name', 'Book Length', 'Average Word Length', 'Length NoRepeat', 'Words', 'least common']) 
    

    # return pd.concat([df, words_df])



def get_features():
    """
    This function is the driver function and takes in no arguments. The purpose of 
    this function is to get features from a book i.e `book length`etc. This function 
    takes in no arguments but will return a dataframe with all the features it has
    collected. First the function calls the book create dictionary function to get 
    all our data of books.
    Then we will do some cleaning and.

    Parameters:
    -----------
    None


    Returns:
    --------
    Pandas Dataframe
        This dataframe constains all the 
    """
    # open all the books in a directory and clean the book of things we dont need
    book_dict = make_dict_of_book("data")

    # create dictionary with book and length
    length_dict = get_length_all(book_dict)

    # crate a dictionary with length of non repeating words
    no_repeat = get_book_length_noreapeat(book_dict)

    # create a dictionary with words that are least/most likely to appear
    least_most_common_words = most_common_words(book_dict)

    # print(least_most_common_words)
    # make a dataframe
    dataframe = to_dataframe(length_dict, book_dict, no_repeat, least_most_common_words)

    return dataframe


get_features()
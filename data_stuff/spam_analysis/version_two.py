import re
PROBS = {}
average_spam = 0

def get_probs_for_words(sms_list:list):
    """
    This function takes in the sms as a list so we can
    check their values in the dictionary. 
    """
    # P(SPAM|word) = P(word|SPAM)*P(SPAM)/ P(word)

    # # probability of word appearing spam
    # P(word|SPAM)

    # # probability of spam happening
    # P(SPAM)

    # # probability of word appearing at all
    # P(word)
    total, ham, spam = get_p_of_label()
    total_spam = 0
    total_ham = 0
    other_words = []
    for i in range(len(sms_list)):
        word = sms_list[i]
        word = str(word)
        if word not in PROBS:
            # If the word is not in the dictionary
            # we ignore it for now
            other_words.append(word)
        else:
            # get values or words
            spam_prob = PROBS[word]['spam']
            ham_prob = PROBS[word]['ham']

            # P(word|label) 
            prob_of_spam_word = spam_prob/spam 
            prob_of_ham_word = ham_prob/ham

            # P(label)
            spam_label = spam/total
            ham_label = ham/total

            # all ocurrances of a word
            # P(word)
            word_prob = spam_prob+ham_prob
            prob_of_spam = word_prob / total

            result_spam = (((prob_of_spam_word)*(spam_label))/(word_prob))
            result_ham = (((prob_of_ham_word)*(ham_label))/(word_prob))
            if result_ham > result_spam:
                total_ham +=1
            elif result_ham < result_spam:
                total_spam +=1
            elif result_ham == result_spam:
                total_spam +=1
                total_ham +=1
    if total_ham > total_spam:
        return ('ham', other_words)
    elif total_ham < total_spam:
        return('spam', other_words)
    elif total_ham == total_spam:
        return ('spam', other_words)


def get_p_of_label():
    """
    Since we are using naive bayes we have to get some 
    probabilities to use in our equation
    """
    total = 0
    spam = 0
    ham = 0
    for key in PROBS:
        word = key
        one_prob = PROBS[word]['spam']
        sec_prob = PROBS[word]['ham']
        spam += one_prob
        ham += sec_prob
        total += one_prob+sec_prob
    return total, ham, spam

#####This function is for setting the data into the probability dictionary   
def assign_vals_to_words(label:str, sms_list:list):
    """
    This function takes in the label of the message and the
    dictionary we created. We then check 
    """
    spam = 'spam'
    ham = 'ham'
    if label == spam:
        # for all spam sms in our train data sets
        for i in range(len(sms_list)):
            # select word
            word = sms_list[i]
            # add 1 to their spam values
            PROBS[word][spam] += 1
    elif label == ham:
        # for all ham sms in our train data sets
        for i in range(len(sms_list)):
            # select word
            word = sms_list[i]
            # add 1 to their ham values
            PROBS[word][ham] += 1       
    else:
        pass


#####This function is for setting the data into the probability dictionary   
def add_words_to_dict(sms_list:list):
    """
    This function takes in a list as its argument and with
    the items of those list we create a dictonary with the
    words as keys. 
    """
    for i in range(len(sms_list)):
        # select current word
        word = sms_list[i]

        # if word is already there ignore it for now
        if word in PROBS:
            pass

        elif word not in PROBS:
            # if word is not in there set their ham and spam values to 1
            PROBS[word] = {'spam' : int(1), 'ham' : int(1)}


####ALL DATA SETS
def clean_string(sms:str) -> list:
    """
    This function takes in a string as its argument and removes
    some stuff such as !{}+()= etc. As well as setting all the 
    words to lowercase. This will help us compare words only 
    and allow the algorithm to prefer better by not getting 
    distracted by irrelavent things. The function returns a list
    of o cleaned lowercase words
    """
    # remove stuff and numbers
    clean = re.sub(r'[\.!#%*()@,:/;"{}+=-]', ' ', sms)
    clean = re.sub(r'[0-9]', ' ', clean)
    # turn string into list and set words to lowercase
    clean_sms = clean.split()
    clean_sms = [token.lower() for token in clean_sms] 
    return clean_sms


def train_func(label:str, sms:str):
    sms_list = clean_string(sms)
    add_words_to_dict(sms_list)
    assign_vals_to_words(label, sms_list)


def test_func(sms:str):
    """
    This is the driver function for actually testing the algorithm.
    What happens first is we send the message trough the function 
    clean_string. This will return a list of words which if the 
    length of that list is 0 or none we return ham. If it is not 
    we keep going and 
    """
    sms_list = clean_string(sms)
    if len(sms_list) == 0:
        return 'ham'
    result, other_words = get_probs_for_words(sms_list)
    if result == 'spam':
        for i in range(len(other_words)):
            word = other_words[i]
            PROBS[word] = {'spam' : int(1), 'ham' : int(1)}
    return result



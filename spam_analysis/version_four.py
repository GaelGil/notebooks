import re
PROBS = {}
average_spam = 0

STOP_WORDS = [ 
    'i',
    'me',
    'my',
    'myself',
    'we',
    'our',
    'ours',
    'ourselves',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves',
    'he',
    'him',
    'his',
    'himself',
    'she',
    'her',
    'hers',
    'herself',
    'it',
    'its',
    'itself',
    'they',
    'them',
    'their',
    'theirs',
    'themselves',
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    'these',
    'those',
    'am',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'do',
    'does',
    'did',
    'doing',
    'a',
    'an',
    'the',
    'and',
    'but',
    'if',
    'or',
    'because',
    'as',
    'until',
    'while',
    'of',
    'at',
    'by',
    'for',
    'with',
    'about',
    'against',
    'between',
    'into',
    'through',
    'during',
    'before',
    'after',
    'above',
    'below',
    'to',
    'from',
    'up',
    'down',
    'in',
    'out',
    'on',
    'off',
    'over',
    'under',
    'again',
    'further',
    'then',
    'once',
    'here',
    'there',
    'when',
    'where',
    'why',
    'how',
    'all',
    'any',
    'both',
    'each',
    'few',
    'more',
    'most',
    'other',
    'some',
    'such',
    'no',
    'nor',
    'not',
    'only',
    'own',
    'same',
    'so',
    'than',
    'too',
    'very',
    's',
    't',
    'can',
    'will',
    'just',
    'don',
    'should',
    'now',]



def get_probs_for_words(sms_list:list):
    """
    This function takes in the sms as a list so we can
    check their values in the dictionary. 
    """
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

            if ham_prob == spam_prob:
                total_spam +=1
                total_ham +=1
            elif ham_prob > spam_prob:
                total_ham +=1
            elif ham_prob < spam_prob:
                total_spam +=1

    return (total_ham, total_spam)

def predict(sms_list:list):
    """
    This function takes in the sms as a list so we can
    check their values in the dictionary. 
    """
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

            if ham_prob == spam_prob:
                total_spam +=1
                total_ham +=1
            elif ham_prob > spam_prob:
                total_ham +=1
            elif ham_prob < spam_prob:
                total_spam +=1

    if total_ham == total_spam:
        return('spam')
    elif total_spam >= 5:
        return('spam')
    else:
        return('ham')




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


def remove_stop_words(sms:list)-> list:
    for i in range(len(sms)):
        if i in STOP_WORDS:
            sms.remove(word)
        else:
            pass

    return sms        


def train_func(label:str, sms:str):
    sms_list = clean_string(sms)
    clean_list = remove_stop_words(sms_list)
    add_words_to_dict(clean_list)
    assign_vals_to_words(label, clean_list)


def driver_func(sms):
    message = clean_string(sms)
    clean_sms = remove_stop_words(message)
    ham, spam = get_probs_for_words(clean_sms)
    return (ham, spam)


def test_func(sms):
    message = clean_string(sms)
    clean_sms = remove_stop_words(message)
    result = predict(clean_sms)
    return result

def get_dict():
    print(PROBS)


def get_sms(sms):
    print(sms)
SPAM_DICT = {
    'you' : 0,
    'won' : 0 ,
    'free' : 0,
    'money' : 0,
    'try' : 0,
    'claim' : 0,
    'prize' : 0,
    'get' : 0,
    'eligable': 0,
    'For': 0,
    'ur' :0, 
    'chance' :0,
    'to' :0,
    'win' :0,
    'a':0,
    '£250':0,
    'wkly':0,
    'shopping':0,
    'spree':0, 
    'TXT:' :0, 
    'SHOP' :0, 
    'to':0, 
    '80878.':0, 
    'Ts&Cs':0, 
    'www.txt-2-shop.com':0, 
    'custcare':0, 
    '08715705022,':0, 
    '1x150p/wk':0
}

def add_to_spam(sms:list, sms_dict:dict):
    """
    If the message is spam it'll append the words
    from the message to a dict. If words are alredy
    in the dict it'll add plus one the their excisting 
    value. If the word is not in the dict it will get 
    added as a key with a value of one.
    """
    for i in range(len(sms)):
        word = sms[i]
        if word in SPAM_DICT:
            previous_val = SPAM_DICT.get(word)
            new_val = previous_val + 5
            SPAM_DICT[i] = new_val 
        else:
            SPAM_DICT[word] = 0 


def spam_or_ham(sms:list, sms_dict:dict):
    """
    This function takes in a list and a dict as its argument.
    We first loop through our dictionary and add up all the 
    values of spam and ham. In the end we compare which value
    is greater to decied weather if the message is spam or not. 
    """
    spam_val = 0 
    ham_val = 0
    for key in sms_dict:
        word = key
        spam_val += sms_dict[word]['spam']
    for key in sms_dict:
        word = key
        ham_val += sms_dict[word]['ham']
    if spam_val > ham_val:
        print('spam')
        add_to_spam(sms, sms_dict)
    elif ham_val > spam_val:
        print('ham')
    else:
        pass


def compare_to_words(sms:list, sms_dict:dict):
    """
    This function takes in a list and dict as its argument and
    compares words in our list to words in SPAM_DICT. If the word
    is found in there we will change the key/word value for that word
    in our dictionay from its previous value of 0 to 1 in spam. 
    If it is not found in there we will change ham to 1
    """
    for key in sms_dict:
        word = key
        if word in SPAM_DICT:
            val = SPAM_DICT.get(key)
            sms_dict[key]['spam'] = val
        elif word not in SPAM_DICT:
            sms_dict[key]['ham'] = 1
        else:
            pass
    return sms, sms_dict 

def crate_dict(sms:str):
    """
    This function takes in a string as its argument and creates
    a nested dictionary with a word as its key and a dictonary 
    with keys set as spam and ham as keys and values set to 0
    """
    sms = sms.split()
    sms = list(sms)
    sms_dict = {}
    for i in range(len(sms)):
        word = sms[i]
        sms_dict[word] = {'spam' : int(0), 'ham' : int(0)}
    return sms, sms_dict


def main_func(message):
    sms_list, sms_dict = crate_dict(message)
    message_list, message_dict = compare_to_words(sms_list, sms_dict)
    spam_or_ham(message_list, message_dict)
    


# text = 'hey are you free tomorrow'
# sms = 'you won free money try out claim prize get free you eligable'
# def get_alg_accuracy(algorithm, training_data) -> float:
#     """
#     This function accepts a spam-detector algorithm and some training data.
#     It compares the predictions made by the algorithm with the actual labels
#     from the dataset. It returns the fraction of predictions that were correct.
#     First the data is transformed into a list of tuples,
#     where the first index in the tuple is the label and the second is the
#     SMS message, i.e. ("ham", "want to hang out tomorrow?")
#     """
#     correct = 0

#     for label, data in training_data:
#         prediction = algorithm(data)

#         if prediction == label:
#             correct += 1

#     fraction_correct = correct / len(training_data)

#     return fraction_correct

# def right_or_wrong(sms:list, sms_dict:dict, result:str, laber:str):
#     """
#     """
#     if result == label:
#         pass
#     elif result != label:
#         add_to_spam(sms, sms_dict)
# main_func(text)

#refrence function above
#NOTES 
#I can add a final function that compares outcome to real label
#if algorithm got it right dont append to the SPAM_DICT but if
#its wrong append to dict. Run alg_v1 then alg_v2 and in the 
# end alg_v3 should start to better detect spam
#use same SPAM_DICT for all functions
SPAM_DICT = {
    'won' : {'spam' : int(1), 'ham' : int(0)},
    'free' : {'spam' : int(1), 'ham' : int(0)},
    'money' : {'spam' : int(1), 'ham' : int(0)},
    'try' : {'spam' : int(1), 'ham' : int(0)},
    'claim' : {'spam' : int(1), 'ham' : int(0)},
    'prize' : {'spam' : int(1), 'ham' : int(0)},
    'get' : {'spam' : int(1), 'ham' : int(0)},
    'eligable': {'spam' : int(1), 'ham' : int(0)},
    'For': {'spam' : int(1), 'ham' : int(0)},
    'ur' :{'spam' : int(1), 'ham' : int(0)}, 
    'chance' :{'spam' : int(1), 'ham' : int(0)},
    'to' :{'spam' : int(1), 'ham' : int(0)},
    'win' :{'spam' : int(1), 'ham' : int(0)},
    'a':{'spam' : int(1), 'ham' : int(0)},
    '£250':{'spam' : int(1), 'ham' : int(0)},
    'wkly':{'spam' : int(1), 'ham' : int(0)},
    'shopping':{'spam' : int(1), 'ham' : int(0)},
    'spree':{'spam' : int(1), 'ham' : int(0)}, 
    'TXT:' :{'spam' : int(1), 'ham' : int(0)}, 
    'SHOP' :{'spam' : int(1), 'ham' : int(0)}, 
    'to':{'spam' : int(1), 'ham' : int(0)}, 
    '80878.':{'spam' : int(1), 'ham' : int(0)}, 
    'Ts&Cs':{'spam' : int(1), 'ham' : int(0)}, 
    'www.txt-2-shop.com':{'spam' : int(1), 'ham' : int(0)}, 
    'custcare':{'spam' : int(1), 'ham' : int(0)}, 
    '08715705022,':{'spam' : int(1), 'ham' : int(0)}, 
    '1x150p/wk':{'spam' : int(1), 'ham' : int(0)},
    'Go': {'spam': 1, 'ham': 0}, 
    'until': {'spam': 1, 'ham': 0},
     'jurong': {'spam': 1, 'ham': 0},
      'point,': {'spam': 1, 'ham': 0}, 
      'crazy..': {'spam': 1, 'ham': 0}, 
    'Available': {'spam': 1, 'ham': 0}, 
    'only': {'spam': 1, 'ham': 0}
}


def add_everything(sms:list):
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
            previous_val = SPAM_DICT[word]['spam']
            new_val = previous_val + 5
            SPAM_DICT[word]['spam'] = new_val
        else:
            SPAM_DICT[word] = {'spam' : int(1), 'ham' : int(0)}


def add_to_spam(sms_dict:dict):
    """
    This function checks if in the nested dictionaries
    the word was set to spam or ham if it is spam we 
    add it to a list and if its not we dont. In the end
    the function returns a list of spam words
    """
    spam_list = []
    for key in sms_dict:
        word = key
        spam_val = sms_dict[word]['spam']
        if spam_val >= 1:
            spam_list.append(word)
    for key in sms_dict:
        word = key
        ham_val = sms_dict[word]['ham']
        if spam_val == 0:
            pass
    # print('SPAM LIST######')
    # print(spam_list)
    return spam_list


def right_or_wrong(sms:list, result:str, label:str):
    """
    This function takes in the message, result of my function,
    and real label of the function. If the result matches the 
    labe everything is cool. If the result does not match the
    """
    if result == label:
        pass
    elif result != label:
        return ('incorrect')


def spam_or_ham(sms:list, sms_dict:dict):
    """
    This function takes in a list and a dict as its argument.
    We first loop through our dictionary and add up all the 
    values of spam and ham. In the end we compare which value
    is greater to decied weather if the message is spam or not. 
    """
    spam_val = 0 
    ham_val = 0
    # for key in sms_dict:
    #     word = key
    #     nested_spam_val = sms_dict[word]['spam']
    #     spam_val += nested_spam_val
    # for key in sms_dict:
    #     word = key
    #     nested_ham_val = sms_dict[word]['ham']
    #     if nested_ham_val == 0:
    #         ham_val += 1
    for key in sms_dict:
        word = key
        nested_spam_val = sms_dict[word]['spam']
        nested_ham_val = sms_dict[word]['ham']
        if nested_ham_val == 1 and nested_spam_val == 0:
            ham_val += 1
        elif nested_ham_val == 0 and nested_spam_val == 1:
            spam_val += 1
    # print('SPAM VAL') 
    # print(spam_val)
    # print('    ')
    # print('HAM_VAL')
    # print(ham_val)
    if spam_val >= ham_val:
        return('spam')
    elif spam_val > ham_val:
        return('spam')
    elif spam_val == ham_val:
        return('spam')
    elif ham_val > spam_val:
        return('ham')


def compare_to_dict(sms:list, sms_dict:dict):
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
            val = SPAM_DICT[word]['spam']
            sms_dict[key]['spam'] = val
        elif word not in SPAM_DICT:
            sms_dict[key]['ham'] = int(1)
        else:
            pass
    # print('AFTER COMPARISON')
    # print(sms)
    # print(sms_dict)
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
        # TODO: make this a defaultdict
        sms_dict[word] = {'spam' : int(0), 'ham' : int(0)}
    # print('ORIGINAL SMS AS LIST AND DICT')
    # print(sms)
    # print(sms_dict)
    # print(' ')
    return sms, sms_dict


def take_some_vals(sms:str, label:str):
    """
    This function is the same as the check function
    except it doest not return anything. It's just to
    populate the dict with some values
    """
    if label == 'spam':
        sms = sms.split()
        sms = list(sms)
        add_everything(sms)


def main_func(message):
    sms_list, sms_dict = crate_dict(message)
    message_list, message_dict = compare_to_dict(sms_list, sms_dict)
    result = spam_or_ham(message_list, message_dict)
    # spam_words = add_to_spam(message_dict)
    # add_everything(spam_words)
    return result


###the issue is that 1 will always be greater than 0

print(SPAM_DICT)
my_str = 'this should not be spam only'
spam_str = 'you won free money HELLL$$$$$$O'

print(main_func(my_str))

print(main_func(spam_str))


print(SPAM_DICT)
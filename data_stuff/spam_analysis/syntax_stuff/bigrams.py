SPAM_DICT = {
    'you won' : 5,
    'free money' : 0,
    'go to' : 0,
    'try out' : 0,
    'claim prize' : 0,
    'get free' : 0,
    'you aligble': 0,}

def add_to_spam(spam:list):
    """
    If the message is spam it'll append the words
    from the message to a dict. If words are alredy
    in the dict it'll add plus one the their excisting 
    value. If the word is not in the dict it will get 
    added as a key with a value of one.
    """
    for i in range(len(spam)):
        word = spam[i]
        if spam[i] in SPAM_DICT:
            previous_val = SPAM_DICT.get(word)
            new_val = previous_val + 1 
            SPAM_DICT[word] = new_val 
        else:
            SPAM_DICT[word] = 0 


def is_spam_or_not(message:list, message_dict:dict):
    """
    This function detemines if message is spam or 
    not based on ...
    """
    spam_num = 0
    for val in message_dict.values():
        spam_num += val
    # print(spam_num)
    if spam_num >= ((len(message)/2) * 10):
        add_to_spam(message)
        print('spam')
    else:
        print('not spam')


def compare_to_phrases(message:list, phrases:dict):
    """
    """
    for key in phrases:
        phrase = key
        # print(phrase)
        if phrase in SPAM_DICT:
            previous_val = phrases.get(phrase)
            new_val = previous_val + 10 
            phrases[phrase] = new_val
        elif phrase not in SPAM_DICT:
            pass
    return message, phrases
            

def get_bigrams(message:str):
    """
    This function takes in a string as its argument and creates a 
    dictionary of bigrams
    """
    # message = message.lower()
    words = message.split()
    words_list =  list(words)
    phrases_dict = {}
    for i in range(len(words_list)-1):
        current_word = words_list[i]
        next_word = words_list[i + 1]
        phrase = current_word + ' ' + next_word
        phrases_dict[phrase] = 0 
    return words_list, phrases_dict


def check(sms):
    words_list, phrases_dict = get_bigrams(sms)
    message, phrases_dict = compare_to_phrases(words_list, phrases_dict)
    is_spam_or_not(message, phrases_dict)

# text = 'hey are you free tomorrow'
# sms = 'you won free money go to try out claim prize get free you aligble'

# check(sms)
# text = "this is a sentence", "so is this one"


# words = my_list.split()
# words_list =  list(words)

# for i in range(len(words_list)-1):
#     current_word = words_list[i]
#     next_word = words_list[i + 1]
#     phrase = current_word + ' ' + next_word
#     print(phrase)


# my_dict = {
#     "free": {
#         "spam": 95,
#         "ham": 12},
#     "win": {
#         "spam": 5,
#         "ham": 1},
#     "the": {
#         "spam": 10000, 
#         "ham": 10000}
# }


def multi(sms:str):
    sms = sms.split()
    sms = list(sms)
    my_dict = {}
    for i in range(len(sms)):
        word = sms[i]
        my_dict[word] = {'spam' : int(3), 'ham' : int(2)}
    # print(my_dict)
    # print(' ')
    # my_dict['my']['spam'] = 1
    # print(my_dict)
    # print(my_dict['my']['spam'])
    nested_val = my_dict['my']['spam']
    print(type(nested_val))
    val = 1
    print(val) 
    val += nested_val
    print(val)
    print(nested_val)
    print('  ')
    val = my_dict.get('my')['spam']
    my_dict
    print(val)
            # nested_spam_val = sms_dict[word]['spam']
    




sms = 'my words are here and they will be list'

multi(sms)

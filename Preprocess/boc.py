#coding: utf-8
import pandas as pd
import yaml

import jieba



'''
    按字切分 而不按词切分
    即 BOC(Characteristic) 与 BOW(Word) 的区别

'''
def singleword(train_data,test_data):
    result_train = []
    for words in train_data['WORDS']:
        character_result_train = ""
        words =  words.split()
        for characters in words:
            if len(characters) > 1:
                for character in characters:
                    character_result_train += (character+ u" ")
            else:
                character_result_train += (characters + u" ")

        character_result_train = character_result_train[0:len(character_result_train)-1]
        result_train.append(character_result_train)


    train_data['SINGLE'] = result_train



    result_test = []
    for words in test_data['WORDS']:
        character_result_test = ""
        words =  words.split()
        for characters in words:
            if len(characters) > 1:
                for character in characters:
                    character_result_test += (character+ u" ")
            else:
                character_result_test += (characters + u" ")
        character_result_test = character_result_test[0:len(character_result_test)-1]
        result_test.append(character_result_test)


    test_data['SINGLE'] = result_test

    train_data.to_csv(
            "train.csv",
             sep = '\t',
            encoding = 'utf8',

        )

    test_data.to_csv(
        "test.csv",
        sep='\t',
        encoding='utf8',

    )
    return result_train,result_test



if __name__ == '__main__':
    train_data = pd.read_csv(
        'train_seg.csv',
        sep='\t',
        encoding='utf8',
        header=0
    )

    test_data = pd.read_csv(
        'test_seg.csv',
        sep='\t',
        encoding='utf8',
        header=0
    )

    singleword(train_data,test_data)
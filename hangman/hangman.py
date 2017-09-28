import sys
import os
import random
import pdb

gameWord = None;
outputVal = None;
def read(filename):
    words = open(filename, mode='rt', encoding='utf-8').readlines()
    lineNumber = int(random.uniform(0, len(words)));
    global gameWord
    gameWord = words[lineNumber].rstrip();
    return True

def paint(hangManStatus):
    d = {
    0:"hangman_base.txt",
    1:"hangman_pole.txt",
    2:"hangman_head.txt",
    3:"hangman_body.txt",
    4:"hangman_legs.txt",
    5:"hangman_complete.txt",
    6:"hangman_init.txt",
    7:"hangman_success.txt"
    }
    try:
        hangman = open("hangman_ascii/" + d[hangManStatus], mode='rt', encoding='utf-8')
    except KeyError:
        print("Key doesn't exist")
    print(hangman.read())

def gameLoop():
    round = 0;
    gameStatus = True;
    hangmanStatus = 0;
    guess = '_' * len(gameWord);
    global outputVal
    outputVal = '_' * len(gameWord);
    guessAttempts = [];
    os.system('clear')
    paint(6);
    input("Press Enter to continue")
    while(gameStatus):
        os.system('clear')
        if((validateWord(guess)==False) & (round != 0)):
            hangmanStatus = hangmanStatus + 1;
            if(hangmanStatus==5):
                gameStatus=False
                print(gameWord)
        paint(hangmanStatus)
        print("Attempts: " + str(set(guessAttempts)))
        if(('_' in outputVal)==False):
            os.system('clear')
            print(gameWord)
            gameStatus=False
            paint(7);
        if(gameStatus):
            guess = input("Guess letter or string\n")
            guessAttempts = guessAttempts + list(set(guess));
        round = round + 1;



def validateWord(guess):
    global outputVal
    wordMatch = False;
    for w in list(guess):
        for i, j in enumerate(list(gameWord)):
            if str(j).upper() == str(w).upper():
                outputVal = list(outputVal);
                outputVal[i] = j;
                wordMatch = True;
    outputVal = ''.join(outputVal)
    print(outputVal);
    return wordMatch;



if (__name__ == '__main__'):
    try:
        read(sys.argv[1]);
        gameLoop();
    except FileNotFoundError:
        print("File reading failed!")

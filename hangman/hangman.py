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
    print(gameWord)
    return True

def paint(hangManStatus):
    d = {
    0:"hangman_base.txt",
    1:"hangman_pole.txt",
    2:"hangman_head.txt",
    3:"hangman_body.txt",
    4:"hangman_legs.txt",
    5:"hangman_complete.txt",
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
    guessAttempts = [];
    global outputVal
    outputVal = '_' * len(gameWord);
    while(gameStatus):
        os.system('clear')
        if((validateWord(guess)==False) & (round != 0)):
            hangmanStatus = hangmanStatus + 1;
        paint(hangmanStatus)
        guess = input("Guess letter or string")
        guessAttempts = guessAttempts + list(set(guess));
        round = round + 1;


def validateWord(guess):
    global outputVal
    wordMatch = False;
    for w in list(guess):
        for i, j in enumerate(list(gameWord)):
            if j == w:
                outputVal = list(outputVal);
                outputVal[i] = w;
                wordMatch = True;
    print(''.join(outputVal));
    return wordMatch;



if (__name__ == '__main__'):
    try:
        read(sys.argv[1]);
        gameLoop();
    except FileNotFoundError:
        print("File reading failed!")
    print(paint(0))

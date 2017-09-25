import sys
import os

def read(filename, lineNumber):
    words = open(filename, mode='rt', encoding='utf-8').readlines()
    if(lineNumber <= len(words)-1 & lineNumber >= 0):
        print(words[lineNumber].rstrip())
        return True
    else:
        print("Index out of bounds")
        return False

def paint(hangManLevel):
    d = {
    0:"hangman_base.txt",
    1:"hangman_pole.txt",
    2:"hangman_head.txt",
    3:"hangman_body.txt",
    4:"hangman_legs.txt",
    5:"hangman_complete.txt",
    }
    hangman = open("hangman_ascii/" + d[hangManLevel], mode='rt', encoding='utf-8')
    print(hangman.read())

if (__name__ == '__main__'):
    try:
        read(sys.argv[1], int(sys.argv[2]))
    except FileNotFoundError:
        print("File reading failed!")
    print(paint(0))

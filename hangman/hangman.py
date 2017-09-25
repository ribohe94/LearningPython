import sys

def read(filename):
    words = open(filename, mode='rt', encoding='utf-8')
    print(words.read())

if (__name__ == '__main__'):
    try:
        read(sys.argv[1])
    except FileNotFoundError:
        print("File reading failed!")

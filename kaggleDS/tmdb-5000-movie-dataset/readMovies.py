import sys
import pandas
import matplotlib.pyplot as plt

movieData = None

def readData(fileName, headers = ['budget', 'release_date']):
    global movieData
    movieData = pandas.read_csv(fileName, usecols = headers)
    movieData.time = pandas.to_datetime(movieData['release_date'], format='%Y-%m-%d')
    movieData.plot('release_date', 'budget')

if (__name__ == '__main__'):
    try:
        readData(sys.argv[1]);
    except FileNotFoundError:
        print("File reading failed!")

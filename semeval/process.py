from __future__ import print_function
from csv import reader
import random
import re


def convert_format(in_file):
  csvfile = open(in_file)
  posfile = open("test.pos.txt", "w")
  negfile = open("test.neg.txt", "w")
  neufile = open("test.neu.txt", "w")

  lines = [line for line in reader(csvfile, delimiter='\t')]
  random.shuffle(lines)

  for line in lines:
    if line[0] == "positive":
      print(line[1], file=posfile)
    elif line[0] == "negative": 
      print(line[1], file=negfile)
    else: 
      print(line[1], file=neufile)
  csvfile.close()
  posfile.close()
  negfile.close()
  neufile.close()

convert_format("test.tsv")

from __future__ import print_function
from csv import reader
import random
import re


def convert_format(in_file):
	csvfile = open(in_file)
	posfile = open("train.pos.txt", "w")
	negfile = open("train.neg.txt", "w")
	neufile = open("train.neu.txt", "w")

	lines = [line for line in reader(csvfile)]
	random.shuffle(lines)

	for line in lines:
		if line[0] == "4":
			print(line[5], file=posfile)
		elif line[0] == "0": 
			print(line[5], file=negfile)
		else:
			print(line[5], file=neufile)
	csvfile.close()
	posfile.close()
	negfile.close()
	neufile.close()

def parse_sentences(in_file):
	infile = open(in_file)
	outfile = open("norm." + in_file, "w")
	for line in reader(infile):
		sent = " ".join(re.findall(r"[\w]+[']?[\w]*", line[0].lower()))
		print(sent, file=outfile)
	infile.close()
	outfile.close()

#convert_format("training.csv")
parse_sentences("train.pos.txt")
parse_sentences("train.neg.txt")
#!/bin/python2
import csv, random

class NewsClassifier:
	def __init__(self):
		train_file = open('data/news-train.csv', 'r')
		self.train_reader = csv.reader(train_file)

		test_in_file = open('data/news-test.csv', 'r')
		self.test_reader = csv.reader(test_in_file)

		test_out_file = open('result/news-test.csv', 'w')
		self.test_writer = csv.writer(test_out_file, quoting=csv.QUOTE_ALL)

		# Skip the header row
		self.train_reader.next()
		self.test_reader.next()

		self.train()
		self.preprocess()
		self.predict()

		train_file.close()
		test_in_file.close()
		test_out_file.close()

	def preprocess(self):
		## Your preprocessing goes here
		pass

	def train(self):
		## Your training logic goes here
		pass

	def predict(self):
		self.test_writer.writerow(['title', 'description'])

		for entry in self.test_reader:
			## Your prediction logic goes here
			prediction = random.choice(['World', 'Business', 'Sports', 'Sci/Tech'])
			self.test_writer.writerow([entry[0], entry[1], prediction])

if __name__ == "__main__":
	trainer = NewsClassifier()

import ccxt
from time import time
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.optimize import *
from math import isnan

# Simulated crypto portfolio
class Portfolio():
	def __init__(self, symbols, epsilon, slack, weights, noop=False):
		self.symbols = symbols
		self.epsilon = epsilon
		self.slack = slack
		self.setWeights(weights)
		self.reset()
		self.tradeFee = 0.0005
		#self.tradeFee = 0.0000
		self.tradePer = 1.0 - self.tradeFee
		self.value = 1.0
		self.noop = noop
	
	def printParams(self):
		print('\nPortfolio parameters:')
		print('\tepsilon: ' + str(self.epsilon))
		print('\tslack: ' + str(self.slack))

	# Re-initialize portfolio state
	def reset(self):
		self.weights = [1. / len(self.symbols)] * len(self.symbols)
	
	# Calculate current portfolio value and set portfolio weights
	def updatePortfolio(self, newWeights, prevWeights, prevValue, prevRates, curRates):
		# Calculate current portfolio value
		rateRatios = np.divide(curRates, prevRates)
		prevValues = np.multiply(prevWeights, prevValue)
		currentValues = np.multiply(rateRatios, prevValues)
		currentVal = sum(currentValues)
		#print('Current portfolio value with new rates: ' + str(currentVal))
		
		for i in range(len(newWeights)):
			if isnan(newWeights[i]):
				print('\n\n\n\t\tnewWeights is nan at: ' + str(i) + '\n\n\n')

		for i in range(len(self.weights)):
			if isnan(self.weights[i]):
				print('\n\n\n\t\tself.weights is nan at: ' + str(i) + '\n\n\n')

		# Calculate difference between current and new weights
		weightDelta = np.subtract(newWeights, self.weights)

		for i in range(len(weightDelta)):
			if isnan(weightDelta[i]):
				print('\n\n\n\t\tweightDelta is nan at: ' + str(i) + '\n\n\n')

		if isnan(self.value):
			print('\n\n\n\t\tself.value is nan\n\n\n')
		valueDelta = [(self.value * delta) for delta in weightDelta]

		# Calculate BTC being bought
		buy = self.tradePer * -sum([v if (v < 0) else 0 for v in valueDelta])
		#print('\nBTC bought: ' + str(buy))

		#print('\t\t\tMax value delta: ' + str(max(valueDelta)))
		posValDeltas = {}
		for i in range(len(valueDelta)):
			if valueDelta[i] > 0:
				posValDeltas[i] = valueDelta[i]

		posValDeltaSum = sum(posValDeltas.values())
		posValDeltaPer = np.divide(list(posValDeltas.values()), posValDeltaSum)

		# Calculate actual positive value changes with trade fees
		realPosValDeltas = [per * self.tradePer * buy for per in posValDeltaPer]
		
		# Calculate overall value deltas
		realValDeltas = []
		for val in valueDelta:
			if val <= 0:
				realValDeltas.append(val)
			else:
				realValDeltas.append(realPosValDeltas.pop(0))

		# Calculate new value
		newValues = np.add(currentValues, realValDeltas)
		newValue = sum(newValues)
		#print('Portfolio value after shift: ' + str(newValue))
		self.setValue(newValue)
	
		self.setWeights(np.divide(newValues, newValue))
		#print('\nNew weights:')
		#print(str(list(self.getWeights())) + '\n')

	# Simulate the pamr agent over a set of data and return the final portfolio value
	def simulate(self, fData):	
		for i in range(len(fData) - 1):
			# Get market relative price vector
			prevRates = []
			curRates = []
			x = []
			values = []

			for j in range(len(symbols)):
				lastClose = fData[i+1][j]
				prevClose = fData[i][j]
				if isnan(lastClose):
					print('\n\n\n\t\tlast close is nan at (' + str(i) + ', ' + str(j) + ')\n\n\n')
				if isnan(prevClose):
					print('\n\n\n\t\tprev close is nan at (' + str(i) + ', ' + str(j) + ')\n\n\n')
				curRates.append(lastClose)
				prevRates.append(prevClose)
				if isnan(lastClose / prevClose):
					print('\n\n\n\t\tx is nan at (' + str(i) + ', ' + str(j) + ')\n\n\n')
				x.append(lastClose / prevClose)
				values.append(self.getWeights()[j] * self.getValue() * x[j])


	
			prevValue = self.getValue()
			self.setValue(sum(values))

			b = np.divide(values, self.getValue())
			prevWeights = self.getWeights()
			
			if not self.noop:
				for j in range(len(b)):
					if isnan(b[j]):
						print('\n\n\n\t\tb is nan at: (' + str(i) + ', ' + str(j) + ')\n\n\n')
				self.setWeights(b)

			# Calculate loss
			loss = max(0, np.dot(b, x) - self.getEpsilon())
			if isnan(loss):
				print('\n\n\n\t\tloss is nan at (' + str(i) + ')\n\n\n')
				

			# Calculate Tau (update step size)
			tau = loss / ((norm(np.subtract(x, np.mean(x))) ** 2) + (1 / (2. * self.getSlack()))) 

			if isnan(tau):
				print('\n\n\n\t\ttau is nan at (' + str(i) + ')\n\n\n')

			# Calculate new portfolio weights
			b = np.subtract(self.getWeights(), np.multiply(tau, np.subtract(x, np.mean(x))))
			for j in range(len(b)):
				if isnan(b[j]):
					print('\n\n\n\t\tb is nan at (' + str(i) + ', ' + str(j) + ')\n\n\n')
			
		#	print('b: ' + str(b))			

			# Project portfolio into simplex domain
			result = minimize(lambda q: norm(np.subtract(q, b)) ** 2, [1. / len(b) for z in b], method='SLSQP', bounds=[(0.0, 1.0) for z in b], constraints={'type': 'eq', 'fun': lambda q: sum(q) - 1.0})
			
			if not self.noop:
			#	print('Result: ' + str(result['x']))
				for j in range(len(result['x'])):
					if isnan(result['x'][j]):
						print('\n\n\n\t\tresult[\'x\'] is nan at (' + str(i) + ', ' + str(j) + ')\n\n\n')
				self.updatePortfolio(result['x'], prevWeights, prevValue, prevRates, curRates) 
		print('\nFinal weights: ' + str(self.getWeights()) + '\n')
		return self.getValue()

	def getEpsilon(self):
		return self.epsilon

	def getSlack(self):
		return self.slack

	def getValue(self):
		return self.value

	def getWeights(self):
		return self.weights[:]

	def setValue(self, value):
		self.value = value

	# Assign new portfolio weights
	def setWeights(self, weights):
		self.weights = weights[:]


def getData(b, t, depth, symbol):
	print('\nGetting data for symbol: ' + symbol)
	data = []
	while True:
		ohlcv = b.fetch_ohlcv(symbol, since=t)
		data = ohlcv[:] if len(data) == 0 else np.concatenate((ohlcv, data))
		t -= 500 * 60000
		if len(data) >= depth:
			break
	return data

def validateTimesteps(data):
	print('Final data len: ' + str(len(data)))
	timesteps = [x[0] for x in data]
	for i in range(len(timesteps) - 1):
		if timesteps[i + 1] - timesteps[i] != 60000:
			plt.plot(timesteps, color='r')
			plt.show()
			return False
	return True

def truncateData(data):
	truncLength = min([len(sym) for sym in data])
	print('Truncated data len: ' + str(truncLength))
	#return [x[:truncLength] for x in data]
	return [x[len(x) - truncLength:] for x in data]

def checkTruncData(data):
	timestamp = data[0][0][0]
	for i in range(1, len(data)):
		if data[i][0][0] != timestamp:
			print('\t\t\tTimestamps not synchronized!')
			return
	print('\t\t\tTimestamps synchronized')

def formatData(data):
	# (symbols x timesteps x features) --> (timesteps x symbols)
	fData = []
	for i in range(len(data[0]) - 1):
		stepData = []
		for j in range(len(data)):
			stepData.append(data[j][i][4])
		fData.append(stepData)
	return fData


now = int(time() * 1000)
start = now - 500 * 60000
binance = ccxt.binance()
binance.load_markets()
symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC', 'NEO/BTC']
#depth = 140000
depth = 110000

print('\nPortfolio symbols: ' + str(symbols))

data = []
for sym in symbols:
	symData = getData(binance, start, depth, sym)
	symData = symData[:-10000]
	valid = validateTimesteps(symData)
	if valid:
		print('Data valid for ' + sym)
	else:
		print('Data invalid for ' + sym + '!!!')
	data.append(symData)
	
tData = truncateData(data)
checkTruncData(tData)
fData = formatData(tData)


print('\n\n' + str(np.array(fData).shape))
b = [1 / float(len(symbols))] * len(symbols)

# Initialize simulated portfolio
port0 = Portfolio(symbols, 0.65, 5, b)
port1 = Portfolio(symbols, 0.75, 5, b)
port2 = Portfolio(symbols, 0.85, 5, b)
port3 = Portfolio(symbols, 0.95, 5, b)
port4 = Portfolio(symbols, 0.65, 4, b)
port5 = Portfolio(symbols, 0.75, 4, b)
port6 = Portfolio(symbols, 0.85, 4, b)
port7 = Portfolio(symbols, 0.95, 4, b)
port8 = Portfolio(symbols, 0.65, 3, b)
port9 = Portfolio(symbols, 0.75, 3, b)
port10 = Portfolio(symbols, 0.85, 3, b)
port11 = Portfolio(symbols, 0.95, 3, b)
ports = [port0, port1, port2, port3, port4, port5, port6, port7, port8, port9, port10, port11]
bh = Portfolio(symbols, 0.95, 3, b, noop=True)

print('\nBuy & Hold:')
val = bh.simulate(fData)
print('\tPortfolio value: ' + str(val) + '\n')

for port in ports:
	port.printParams()
	print('---Simulating portfolio---')
	val = port.simulate(fData)
	print('\tPortfolio value: ' + str(val) + '\n')


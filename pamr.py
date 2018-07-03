import ccxt
from time import time
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.optimize import *
from math import isnan

# Simulated crypto portfolio
class Portfolio():
	def __init__(self, symbols, epsilon, slack, interval, weights, noop=False):
		self.symbols = symbols
		self.epsilon = epsilon
		self.slack = slack
		self.interval = interval
		self.setWeights(weights)
		self.reset()
		self.tradeFee = 0.0005
		#self.tradeFee = 0.0000
		self.tradePer = 1.0 - self.tradeFee
		self.value = 1.0
		self.values = [1.0]
		self.noop = noop
	
	def printParams(self):
		print('\nPortfolio parameters:')
		print('\tepsilon: ' + str(self.epsilon))
		print('\tslack: ' + str(self.slack))
		print('\tinterval: ' + str(self.interval))

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
		
		# Calculate difference between current and new weights
		weightDelta = np.subtract(newWeights, self.weights)

		valueDelta = [(self.value * delta) for delta in weightDelta]

		# Calculate BTC being bought
		buy = self.tradePer * -sum([v if (v < 0) else 0 for v in valueDelta])

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
		self.setValue(newValue)
	
		self.setWeights(np.divide(newValues, newValue))

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
				curRates.append(lastClose)
				prevRates.append(prevClose)
				x.append(lastClose / prevClose)
				values.append(self.getWeights()[j] * self.getValue() * x[j])
	
			prevValue = self.getValue()
			self.setValue(sum(values))
			self.values.append(self.getValue())

			b = np.divide(values, self.getValue())
			prevWeights = self.getWeights()
			
			self.setWeights(b)
		
			# Redistribute portfolio on an interval
			if (not self.noop) and (i % self.interval == 0):
				# Calculate loss
				loss = max(0, np.dot(b, x) - self.getEpsilon())

				# Calculate Tau (update step size)
				tau = loss / ((norm(np.subtract(x, np.mean(x))) ** 2) + (1 / (2. * self.getSlack()))) 

				# Calculate new portfolio weights
				b = np.subtract(self.getWeights(), np.multiply(tau, np.subtract(x, np.mean(x))))
			
				# Project portfolio into simplex domain
				result = minimize(lambda q: norm(np.subtract(q, b)) ** 2, [1. / len(b) for z in b], method='SLSQP', bounds=[(0.0, 1.0) for z in b], constraints={'type': 'eq', 'fun': lambda q: sum(q) - 1.0})
			
				# Update portfolio with projected new weights
				self.updatePortfolio(result['x'], prevWeights, prevValue, prevRates, curRates) 
		print('\n\tFinal Weights: ' + str(np.array(self.getWeights())) + '\n') 
		return self.getValue()

	def getEpsilon(self):
		return self.epsilon

	def getSlack(self):
		return self.slack

	def getValue(self):
		return self.value

	def getWeights(self):
		return self.weights[:]

	def getValues(self):
		return self.values

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

# Make sure there are no gaps or repeats in a sequence of timesteps
def validateTimesteps(data):
	print('Final data len: ' + str(len(data)))
	timesteps = [x[0] for x in data]
	for i in range(len(timesteps) - 1):
		if timesteps[i + 1] - timesteps[i] != 60000:
			plt.plot(timesteps, color='r')
			plt.show()
			return False
	return True

# Trim the starts of sequences to ensure similar length for all symbols
def truncateData(data):
	truncLength = min([len(sym) for sym in data])
	print('Truncated data len: ' + str(truncLength))
	#return [x[:truncLength] for x in data]
	return [x[len(x) - truncLength:] for x in data]

# Ensures the start of each sequence is the same after being truncated
def checkTruncData(data):
	timestamp = data[0][0][0]
	for i in range(1, len(data)):
		if data[i][0][0] != timestamp:
			print('\t\t\tTimestamps not synchronized!')
			return
	print('\t\t\tTimestamps synchronized')

# Reformat the data into a single multivariate sequence
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
symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC']
#symbols = ['TRX/BTC', 'ETC/BTC', 'BCH/BTC', 'IOTA/BTC', 'ZRX/BTC', 'WAN/BTC', 'WAVES/BTC', 'SNT/BTC', 'MCO/BTC', 'DASH/BTC', 'ELF/BTC', 'AION/BTC', 'STRAT/BTC', 'XVG/BTC', 'EDO/BTC', 'IOST/BTC', 'WABI/BTC', 'SUB/BTC', 'OMG/BTC', 'WTC/BTC', 'LSK/BTC', 'ZEC/BTC', 'STEEM/BTC', 'QSP/BTC', 'SALT/BTC', 'ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC']
#symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC']
#depth = 110000
depth = 140000

print('\nPortfolio symbols: ' + str(symbols))
print('Managing ' + str(len(symbols)) + ' cryptocurrencies in each portfolio')

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
port0 = Portfolio(symbols, 0.25, 7, 1, b)
port1 = Portfolio(symbols, 0.25, 7, 5, b)
port2 = Portfolio(symbols, 0.35, 7, 1, b)
port3 = Portfolio(symbols, 0.35, 7, 5, b)
port4 = Portfolio(symbols, 0.45, 7, 1, b)
port5 = Portfolio(symbols, 0.45, 7, 5, b)
port6 = Portfolio(symbols, 0.55, 7, 1, b)
port7 = Portfolio(symbols, 0.55, 7, 5, b)

ports = [port0, port1, port2, port3, port4, port5, port6, port7]
bh = Portfolio(symbols, 0.95, 3, 1, b, noop=True)

print('\nBuy & Hold:')
val = bh.simulate(fData)
print('\tPortfolio value: ' + str(val) + '\n')

for port in ports:
	port.printParams()
	print('---Simulating portfolio---')
	val = port.simulate(fData)
	print('\tPortfolio value: ' + str(val) + '\n')

plt.title('Portfolio value vs minutes')
plt.plot(bh.getValues(), label='Buy & Hold', color='#000000')
plt.plot(port0.getValues(), label='Portfolio 0', color='#FF0000')
plt.plot(port1.getValues(), label='Portfolio 1', color='#FF9000')
plt.plot(port2.getValues(), label='Portfolio 2', color='#FFFF00')
plt.plot(port3.getValues(), label='Portfolio 3', color='#00FF00')
plt.plot(port4.getValues(), label='Portfolio 4', color='#00D8FF')
plt.plot(port5.getValues(), label='Portfolio 5', color='#0000FF')
plt.plot(port6.getValues(), label='Portfolio 6', color='#9800FF')
plt.plot(port7.getValues(), label='Portfolio 7', color='#FA00FF')
plt.legend()
plt.show()

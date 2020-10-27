from dataset import read_train_sets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.full1 = nn.Linear(in_features=12 * 22 * 22, out_features=120)
        self.full2 = nn.Linear(in_features=120, out_features=60)
        self.output = nn.Linear(in_features=60, out_features = 2)
    
    def forward(self, input):
        #move through the first convolution layer
        input = self.convolution1(input)
        input = F.relu(input)
        input = F.max_pool2d(input, kernel_size=2, stride=2)
        
        #move through the second convolution layer
        input = self.convolution2(input)
        input = F.relu(input)
        input = F.max_pool2d(input, kernel_size=2, stride=2)
        
        #move through the first linear layer
        input = input.reshape(-1, 12 * 22 * 22)
        input = self.full1(input)
        input = F.relu(input)
        
        #move through the second linear layer
        input = self.full2(input)
        input = F.relu(input)
        
        #move to output layer
        input = self.output(input)
        #input = F.relu(input)
        
        return input

#check the predition against the class label    
def predictionCheck(prediction, label):
    if(label[0] == 1 and prediction[0] > prediction[1]):
        return 1
    elif(label[1] == 1 and prediction[0] < prediction[1]):
        return 1
    else:
        return 0

#create the network
network = network()

#train the network
trainData = read_train_sets('data/training_data/', 100, ['pembroke', 'cardigan'], 0.2)
optimizer = optim.Adam(network.parameters(), lr = 0.0001)
print("Training Network...")

#train the network for 10 epoch
for epoch in range(10):
    totalLoss = 0
    totalCorrect = 0
    
    #run through the entire training set
    for index in range(len(trainData.train.images())):
        sample = trainData.train.images()[index].T
        sample = torch.FloatTensor(sample)
        sample = sample.unsqueeze(0)
        label = torch.FloatTensor(trainData.train.labels()[index])
        
        #make prediction
        prediction = network(sample)[0]
        
        #calculate loss
        loss = F.binary_cross_entropy_with_logits(prediction, label)
        
        #back propagate update edge weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #accumulate the total loss and correct predictions
        totalLoss += loss.item()
        totalCorrect += predictionCheck(prediction, label)
        index += 1
    #print epoch results to console   
    print("Epoch: " + str(epoch) + ", Total Correct: " + str(totalCorrect) + ", Total Loss: " + str(totalLoss))


#run the network on the validation data
print()
print("Testing Validation Data...")
validationData = trainData.valid.images()
validationLabels = trainData.valid.labels()

totalCorrect = 0
total = len(validationData)
#run on each validation input
for index in range(total):
    #get input from validation set
    sample = validationData[index].T
    sample = torch.FloatTensor(sample)
    sample = sample.unsqueeze(0)
    label = torch.FloatTensor(validationLabels[index])
    
    #make prediction
    prediction = network(sample)[0]
    
    #accumulate total correct predictions
    totalCorrect += predictionCheck(prediction, label)
validationPercent = (totalCorrect / total) * 100
print("Validation Percent Correct: " + str(validationPercent) + "%")


#run the network on the test data
print()
print("Testing Test Data...")
#import training data and labels
data = read_train_sets('data/testing_data/', 100, ['pembroke', 'cardigan'], 0)
testData = data.train.images()
testLabels = data.train.labels()

totalCorrect = 0
total = len(testData)
#run on each testing input
for index in range(total):
    #get input from testing set
    sample = testData[index].T
    sample = torch.FloatTensor(sample)
    sample = sample.unsqueeze(0)
    label = torch.FloatTensor(testLabels[index])
    
    #make prediction
    prediction = network(sample)[0]
    
    #accumulate total correct responses
    totalCorrect += predictionCheck(prediction, label)
testPercent = (totalCorrect / total) * 100
print("Test Percent Correct: " + str(testPercent) + "%")
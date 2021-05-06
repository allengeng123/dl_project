# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:26 2021

@author: longq
"""


import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import GRU,  Embedding,  Dropout,  Dense, Activation, TimeDistributed
# reference: "https://github.com/gauravtheP/Music-Generation-Using-Deep-Learning/blob/master/Complete_Code"

#data conversion
file = open("Data.txt", mode = 'r')
data = file.read()
file.close()

set_data = sorted(list(set(data)))
mapping_char_idx = dict()
mapping_idx_char = dict()

for (i, ch) in enumerate(set_data):
    mapping_char_idx[ch] = i 
    mapping_idx_char[i] = ch
num_char = len(mapping_char_idx)



#data processing
input_long_seq = []
for d in data:
    input_long_seq.append(mapping_char_idx[d])
input_long_seq = np.array(input_long_seq, dtype="float32")

b_size = 64
seq_size = 64

b_num = int(input_long_seq.shape[0]/b_size)

def batch_generator(input_long_seq, num_char, b_num, b_size, seq_size):
    for s in range(0, b_num - seq_size, seq_size): 
        x = np.zeros((b_size, seq_size)) 
        y = np.zeros((b_size, seq_size, num_char))  
        for bi in range(0, b_size):  
            for i in range(0, seq_size): 
                idx_ = int(b_num * bi + s + i)
                x[bi, i] = input_long_seq[idx_]
                y[bi, i, int(input_long_seq[idx_+1])] = 1 
        yield x, y



#model 1
model1 = Sequential()
model1.add(Embedding(input_dim = num_char, output_dim = 256, batch_input_shape = (b_size, seq_size))) 
model1.add(GRU(128, return_sequences = True, stateful = True))
model1.add(TimeDistributed(Dense(num_char)))
model1.add(Activation("softmax"))
    
model1.summary()
model1.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])




#model 2
model2 = Sequential()
model2.add(Embedding(input_dim = num_char, output_dim = 256, batch_input_shape = (b_size, seq_size))) 
model2.add(GRU(128, return_sequences = True, stateful = True))
model2.add(GRU(128, return_sequences = True, stateful = True))
model2.add(TimeDistributed(Dense(num_char)))
model2.add(Activation("softmax"))
    
model2.summary()
model2.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])



#model 3
model3 = Sequential()
model3.add(Embedding(input_dim = num_char, output_dim = 512, batch_input_shape = (b_size, seq_size))) 
model3.add(GRU(256, return_sequences = True, stateful = True))
model3.add(Dropout(0.3))
model3.add(GRU(256, return_sequences = True, stateful = True))
model3.add(Dropout(0.3))
model3.add(TimeDistributed(Dense(num_char)))
model3.add(Activation("softmax"))
    
model3.summary()
model3.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])



#training

num_epochs = 20
verbose = True


print("training model1")
loss1 = []
accuracy1 = []
for ep in range(num_epochs):
    if verbose:
        print("Epoch {}".format(ep+1))
    for (i, d) in enumerate(batch_generator(input_long_seq, num_char, b_num, b_size, seq_size)):
        x, y = d
        loss_, accuracy_ = model1.train_on_batch(x, y) 
        if verbose:
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, loss_, accuracy_))
    loss1.append(loss_)
    accuracy1.append(accuracy_)
  
    
print("training model2")
loss2 = []
accuracy2 = []
for ep in range(num_epochs):
    if verbose:
        print("Epoch {}".format(ep+1))
    for (i, d) in enumerate(batch_generator(input_long_seq, num_char, b_num, b_size, seq_size)):
        x, y = d
        loss_, accuracy_ = model2.train_on_batch(x, y) 
        if verbose:
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, loss_, accuracy_))
    loss2.append(loss_)
    accuracy2.append(accuracy_)
    
print("training model3")
loss3 = []
accuracy3 = []
for ep in range(num_epochs):
    if verbose:
        print("Epoch {}".format(ep+1))
    for (i, d) in enumerate(batch_generator(input_long_seq, num_char, b_num, b_size, seq_size)):
        x, y = d
        loss_, accuracy_ = model3.train_on_batch(x, y) 
        if verbose:
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, loss_, accuracy_))
    loss3.append(loss_)
    accuracy3.append(accuracy_)
 
    
#save the model
save_model = True    
if save_model:
    model_json = model3.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        model3.save_weights("model.h5")

 


#plot of the results
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6)) 
fig.suptitle('Music RNN Training Results')
ax1.plot(loss1, label='model1')
ax1.plot(loss2, label='model2')
ax1.plot(loss3, label='model3')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend(loc='best')
ax2.plot(accuracy1, label='model1')
ax2.plot(accuracy2, label='model2')
ax2.plot(accuracy3, label='model3')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.legend(loc='best')
plt.savefig('Music RNN Training Results')
    



#create prediction model and load weights
pred_model = Sequential()
pred_model.add(Embedding(input_dim = num_char, output_dim = 512, batch_input_shape = (1, 1)))   
pred_model.add(GRU(256, return_sequences = True, stateful = True))
pred_model.add(Dropout(0.3))
pred_model.add(GRU(256, stateful = True)) 
pred_model.add(Dropout(0.3))
pred_model.add((Dense(num_char)))
pred_model.add(Activation("softmax"))

pred_model.load_weights("model.h5")




#music generation
init_idx = np.random.randint(0, num_char)
generation_len = np.random.randint(250, 500)


generation_seq = []
generation_seq.append(init_idx)


for g in range(generation_len):
    generation_char = np.zeros((1, 1))
    generation_char[0, 0] = generation_seq[-1]  
    prob = pred_model.predict_on_batch(generation_char)    
    res = np.random.choice(range(num_char), p = prob.reshape(-1)) 
    generation_seq.append(res)
    
seq = ''
for s in generation_seq:
    seq += mapping_idx_char[s]
   
    
    

# keep generated seqence only between chars "\n" and "\n\n\n" 
ctr = 0
flag_ctr = 0
start = 0
end = 0
for i in seq:
    ctr += 1
    if i == "\n" and flag_ctr == 0:
        start = ctr 
        flag_ctr = 1
        continue
    if i == "\n" and seq[ctr] == "\n" and flag_ctr == 1:
        end = ctr-1
        break
final_seq = seq[start:end]
print(final_seq)
print(seq)



# save music 
file1 = open("generated_music.txt","w+")
file1.write(seq)
file1.close()

file2 = open("generated_fine_music.txt","w+")
file2.write(final_seq)
file2.close()


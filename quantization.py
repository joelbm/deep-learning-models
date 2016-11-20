
import copy

# gets the model weights and saves them as a back up.
def back_up():
  all_weights_bu = get_weights()
  return all_weights_bu

# Gathers the model's trained weights.
def get_weights():
  # Weight gathering
  all_weights = []
  for layer in model.layers:
    all_weights.append(layer.get_weights())
  return all_weights 

# Quantizes weights by adding an error of the size of the quantiztion step.
# Quantizes each layer with its own reference which is the max value of the layer, 
# other options could be quantizing with 1 as the reference but the one used is more efficient since
# uses the complete range [-max, max] and therefore the quantization step is smaller.
def q_weights(nb_wq, all_weights_bu, qsf):
  # Quantization
  # weight quantization per layer
  all_weightsq = copy.deepcopy(all_weights_bu)
  #max_value = np.max(vectorize(all_weightsq)) # Quantize all layers with the same reference.
  qsf_cnt = 0
  for i in range(len(all_weightsq)):
    if (all_weightsq[i] != [] and qsf <= qsf_cnt):
      # Weights are in [0]
      max_value = np.max((all_weightsq[i][0])) # Quantize each layer with its own reference.
      all_weightsq[i][0] =  np.int32(all_weightsq[i][0]/max_value*2**nb_wq[i])*max_value/2**nb_wq[i]
      # Biases in [1]
      max_value = np.max((all_weightsq[i][1])) # Quantize each layers with its own reference.
      all_weightsq[i][1] =  np.int32(all_weightsq[i][1]/max_value*2**nb_wq[i])*max_value/2**nb_wq[i]
    qsf_cnt += 1 
  # Setting back quantized weights
  i = 0
  for layer in modelq.layers:
    layer.set_weights(all_weightsq[i])
    i += 1
  return all_weightsq

# Gets the minimum size of an accumulator with the assumptions that:
# each layer is serially calculated with an ALU.
# @@@ Bias is not added, TBI
def accu_size(layer, nb_iq, nb_wq):
  accu_size = 0
  max_accu_size = 0
  for i in range(len(layer)):
    accu_size += layer[i]
    #print accu_size
    if accu_size > max_accu_size:
      max_accu_size = accu_size
  return np.log2((np.abs(max_accu_size)+1)*2**(nb_wq+nb_iq)) # +2 is due to the sign (+1) and additional guardband (+1).

# Gets accu_size for all layers.
def accu_sizes(all_weights, nb_iq, nb_wq):
  for i in range(len(all_weights)):
    if (all_weights[i] != []):
      size = accu_size(all_weights[i][0].reshape(-1), nb_iq, nb_wq)
    print("Accu size for layer", i, "    : ", size)

# Returns the number of weights in a model.
def nb_weights(all_weights):
  nb_weights_total = 0
  nb_weights_layer = 0
  for i in range(len(all_weights)):
    if (all_weights[i] != []):
      nb_weights_layer = 1
      for dim in np.shape(all_weights[i][0]): nb_weights_layer *= dim
    nb_weights_total += nb_weights_layer
  return nb_weights_total

# Returns the number of biases in a model
def nb_biases(all_weights):
  nb_biases_total = 0
  nb_biases_layer = 0
  for i in range(len(all_weights)):
    if (all_weights[i] != []):
      nb_biases_layer = 1
      for dim in np.shape(all_weights[i][1]): nb_biases_layer *= dim
    nb_biases_total += nb_biases_layer
  return nb_biases_total

# Returns the number of weights and biases
def nb_weibias(all_weights):
  nb_weibias_total = nb_biases(all_weights) + nb_weights(all_weights)
  return nb_weibias_total

# Returns the number of weights == 0 in a model.
def cnz(all_weights):
  z = 0
  for i in range(len(all_weights)):
    if (all_weights[i] != []):
      for j in range(len(all_weights[i])):
         if (all_weights[i][j] != []):
           for k in range(len(all_weights[i][j])):
             z += np.count_nonzero(all_weights[i][j][k])
  return z

def plot_tf(line):
  import matplotlib.pyplot as plt
  plt.plot(line)
  plt.show()

# Creates a dictionary with all the different weights in a model as words.
def create_dict(all_weights):
  dict = []
  for i in range(len(all_weights)):
    if (all_weights[i] != []):
      dict = np.hstack((dict,np.unique(all_weights[i][0])))
      dict = np.hstack((dict,np.unique(all_weights[i][1])))
      #print np.shape(np.unique(all_weights[i][j][k].reshape(-1))) 
  dict = np.unique(dict)
  return dict

# Does Run Length Encoding of only zeros.
def RLE_zeros(line):
  new_line = []
  cnt = 0
  for i in range(len(line)):
    if 0 == line[i]:      # If they are the same keep counting
      if i == len(line)-1:      # For the last element count but store too.
        cnt += 1
        new_line.append(cnt)
      else:
        cnt += 1
    else:                       # If they are not the same
      if cnt != 0:              # But they were...
        new_line.append(cnt)    # store the count
        new_line.append(line[i])# but also the new value...
        cnt = 0
      else:
        new_line.append(line[i])# Otherwise, store the value not yet coded.
  return new_line

# Does Run Lenght Encoding of all zeros in all weights.
def RLE_zeros_all(all_weightsq):
  # nb of encoded weights
  nb_eweights = 0
  for i in range(len(all_weightsq)):
    if all_weightsq[i] != []:
      nb_eweights += np.shape(RLE_zeros(all_weightsq[i][0].reshape(-1)))[0]
  # nb of encoded biases
  nb_ebiases = 0
  for i in range(len(all_weightsq)):
    if all_weightsq[i] != []:
      nb_ebiases += np.shape(RLE_zeros(all_weightsq[i][1].reshape(-1)))[0]
  # The following is not correct... but left it for now, since we will be always in a better case.
  # Number of bits depend on number of different elements, some layers will share some elements
  # but I don take it here into account.
  nb_bits = 0
  for i in range(len(all_weightsq)):
    if all_weightsq[i] != []:
      nb_bits += len(np.unique(RLE_zeros(all_weightsq[i][0].reshape(-1))))
  nb_bits = np.log2(nb_bits)
  print("Number of encoded weights : ", nb_eweights)
  print("Number of encoded biases  : ", nb_ebiases)
  print("Number of bits per weight : ", nb_bits)
  print("Memory in KB              : ", (nb_eweights+nb_ebiases)*np.ceil(nb_bits)/float(1024))
  return nb_eweights, nb_ebiases, nb_bits

# Returns the input values quantized.
def q_inputs(X, nb_iq):
  X_q = np.int32(np.dot(X,2**nb_iq))/float(2**nb_iq)
  return X_q

# Does Run Length Encoding of line for value, coded dictionary starts from start
def RLE(line,value,start):
  new_line = []
  cnt = start
  for i in range(len(line)):
    if value == line[i]:          # If they are the same keep counting
      if i == len(line)-1:          # For the last element count but store too.
        cnt += 1
        new_line.append(cnt)
      else:
        cnt += 1
    else:                         # If they are not the same
      if cnt != start:              # But they were...
        new_line.append(cnt)          # store the count
        new_line.append(line[i])      # and also the new value...
        cnt = start
      else:
        new_line.append(line[i])    # Otherwise, store the value not yet coded.
  return new_line, max(new_line)

# Does the complete RLE.
def RLE_all(line, dict):
  line_rled = line
  start = 0
  for i in range(len(dict)):
    line_rled,start = RLE(line_rled, dict[i], start)
  dict_size = np.shape(np.unique(line_rled))[0]
  print("Dictionary encoded size   : ", dict_size) 
  print("Number of bits per symbol : ", np.log2(dict_size))
  print("Memory positions needed   : ", np.shape(line_rled)[0])
  print("Memory in KB              : ", np.log2(np.shape(np.unique(line_rled))[0])*np.shape(line_rled)[0]/float(1024))
  return line_rled

# Returns a vector of weights and biases.
def vectorize(all_weightsq):
  vector = []
  for i in range(len(all_weightsq)):
    if all_weightsq[i] != []:
      vector = np.append(vector, all_weightsq[i][0].reshape(-1))
      vector = np.append(vector, all_weightsq[i][1].reshape(-1))
  return vector

# Quantizes the model with nb_wq bits for weights and np_iq bits for the inputs.
def q_model(nb_wq, nb_iq):
  mem_wasted(32)
  # Input quantization
  X_testq = q_inputs(X_test, nb_iq)
  # Weight gathering
  # all_weights = all_weights_bu # Needs to be run: all_weights_bu = back_up()
  # Weight quantization and setting back to the model.
  all_weightsq = q_weights(nb_wq,all_weights_bu, 0)
  # Sizes for accumulators.
  accu_sizes(all_weightsq, nb_wq, nb_iq)
  # RLE applied, obtain number of weighgts and sizes.
  dict = create_dict(all_weightsq)
  weights_vected = vectorize(all_weightsq)
  rle_vected = RLE_all(weights_vected,dict)
  execfile('huffman.py')
  huffman_mem(rle_vected)  
  model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
  score = model.evaluate(X_testq, Y_test, verbose=0)
  print('Test w & i q score      :', score[0])
  print('Test w & i q accuracy   :', score[1])
  return X_testq, all_weightsq, dict, weights_vected, rle_vected

# Memory wasted... just for comparison...
def mem_wasted(bits):
  result = nb_weibias(all_weights_bu)
  print("Memory in Kbits original     : ", result*bits/float(1024))

def name_strip(name):
  if name.endswith('.jpg'):
    name = name[:-9]
    return name

def evaluate_q(predsq, preds):
  result = 0
  predictions  = decode_predictions(preds)
  predictionsq = decode_predictions(predsq)
  for i in range(len(predictions[0])):
    for j in range(len(predictions[0])):
      if predictions[0][i][0] == predictionsq[0][j][0]:
        result += 1
  return result/float(5)*100 

# Returns the number of bits which are 0 and are used to code weights.
def cntz_bits(all_weightsq, nb_wq):
  z = 0
  for i in range(len(all_weightsq)):
    if (all_weightsq[i] != []):
      for j in range(len(all_weightsq[i])):
         if (all_weightsq[i][j] != []):
           for k in range(len(all_weightsq[i][j])):
             z += np.count_nonzero(all_weightsq[i][j][k].reshape(-1))*nb_wq[i]
  return z

def gather_info(nb_wq,):
  all_weights = get_weights()
  all_weightsq = q_weights(nb_wq,all_weights)
  RLE_zeros_all(all_weightsq)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Le codage de Huffman
# Théorie de l'information et du codage
# Etudiant: Boubakr NOUR <n.boubakr@gmail.com>
# Universite Djilali Liabes (UDL) de Sidi Bel Abbes

import heapq
from collections import defaultdict

def encode(frequency):
    heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])    
    return heap
    #return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_mem(rle_vected):
  # Create frequency dictionary
  frequency = defaultdict(int)
  #  From the RLE vectorized and quantized vectors create the frequency vector.
  for each in rle_vected:
    frequency[each] += 1

  # Code each symbol as a huffman code.
  huff = encode(frequency)

  # calculate memory
  huff_bits = np.dot(0,range(len(huff[0]))) # Initializing with 0s and size huff, not very efficient...
  for i in range(1,len(huff[0])):
    for j in range(1,len(huff[0][i][1])):
      huff_bits[i] += 1

  nb_bits = 0  
  for i in range(len(frequency)):
    if frequency.get(i) != None:
      index = huff[0][i][0]
      try:
        nb_bits += frequency.get(index)*huff_bits[index]
      except IndexError:
        nb_bits = 0
  print("Memory in KB RLE + Huffman : ", nb_bits/float(1024))


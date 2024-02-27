from typing import Any, List, Optional, Tuple, overload, Type, Dict
from functools import partial
from scipy import stats
import torchhd, torch
import numpy as np
import math

import sys
sys.path.append('src')
from shared_code.helpers import normalise_partial, theoretical_similarity
from shared_code.classes import Codebook
class Column:
  def __init__(self, name:str, vsa_type:str = 'MAP', elem_dim:int = 10_000, dtype:Type = str):
    self.name = name
    self.dim = elem_dim
    self.vsa_type = vsa_type
    self.atomic_vector: torchhd.VSATensor = torchhd.random(1, self.dim, vsa=self.vsa_type)[0]
    self.dtype = type
    self.codebook: Codebook = Codebook(dim=self.dim, vsa_type=self.vsa_type)

  def add_value(self, value:str):
    return self.codebook.add_value(value)
  
  def __getitem__(self, key):
    return self.codebook[key]
  

class HDDB:
  def __init__(self, dim:int=10_000, vsa_type:str='MAP'):
    self.dim = dim
    self.vsa_type = vsa_type
    self.normalise = partial(normalise_partial, vsa_type)
    self.columns: Dict[str, Column] = {}
    # If we use primary keys, we can use a codebook for rows instead of a list
    # Primary key will then be the value of the row
    self.rows: Codebook = Codebook(dim=self.dim, vsa_type=self.vsa_type)
    # self.rows: List[torchhd.VSATensor] = []

  def similiraity_cutoff(self, certainty:float = 0.9):
    if self.vsa_type != 'BSC':
      raise NotImplementedError
    amount_bundled = len(self.columns)
    exp_sim = theoretical_similarity(amount_bundled, self.vsa_type)
    var = exp_sim * (1-exp_sim) * (1/self.dim)
    c = exp_sim + stats.norm.ppf(1-certainty) * math.sqrt(var)
    return c

  def set_columns(self, columns: Dict[str, Column] | List[str]): #TODO: typing for ints?
    if isinstance(columns[0], str):
      columns = {col: Column(col, vsa_type=self.vsa_type, elem_dim=self.dim) for col in columns}
    self.columns = columns

  def add_row(self, row_key: str, values: List[Any]):
    row_vectors = torchhd.empty(len(self.columns), self.dim, vsa=self.vsa_type)
    for i, col in enumerate(self.columns.values()):
      col_value = col.add_value(values[i])
      col_key_value_pair = torchhd.bind(col.atomic_vector, col_value)
      row_vectors[i] = col_key_value_pair
      # row = torchhd.bundle(row, col_key_value_pair)
    row = torchhd.multiset(row_vectors)
    self.rows.add_value(row_key, self.normalise(row))

  def most_similar_rows(self, row: torchhd.VSATensor, topk: int = 1):
    return self.rows.most_similar_values(row, topk)
      
  def __getitem__(self, key):
    return self.rows[key]
from typing import Any, List, Optional, Tuple, overload, Type, Dict
from utils import Dictionary, Codebook, normalise_partial
from functools import partial
import torchhd, torch

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

def similiraity_cutoff(dim, num_cols, vsa_type):
  

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

  def set_columns(self, columns: Dict[str, Column] | List[str]): #TODO: typing for ints?
    if isinstance(columns[0], str):
      columns = {col: Column(col, vsa_type=self.vsa_type, elem_dim=self.dim) for col in columns}
    self.columns = columns

  def add_row(self, row_key: str, values: List[Any]):
    row = torchhd.empty(1, self.dim, vsa=self.vsa_type)[0]
    for i, col in enumerate(self.columns.values()):
      col_value = col.codebook.add_value(values[i])
      col_key_value_pair = torchhd.bind(col.atomic_vector, col_value)
      row = torchhd.bundle(row, col_key_value_pair)
    self.rows.add_value(row_key, self.normalise(row))

  def most_similar_rows(self, row: torchhd.VSATensor, topk: int = 1):
    return self.rows.most_similar_values(row, topk)
      
  def __getitem__(self, key):
    return self.rows[key]
  
      

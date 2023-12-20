from abc import ABC, abstractmethod

class AbstractTableParser(ABC):
    """Abstract class for parsing tabular data"""
    @abstractmethod
    def parse(self, data):
      """Parse tabular data 
      Args:
        data: tabular data
      Returns:
        Tuple of (header, rows)
      """
      pass


class CSVParser(AbstractTableParser):
    """CSV parser"""
    def __init__(self, delimiter=','):
        self.delimiter = delimiter

    def parse(self, data):
        """Parse CSV data 
        Args:
          data: CSV data
        Returns:
          Tuple of (header, rows)
        """
        rows = []
        for line in data.split('\n'):
            rows.append(line.split(self.delimiter))
        return rows[0], rows[1:]
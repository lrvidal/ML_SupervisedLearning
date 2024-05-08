import numpy as np

class FileParser:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.parse()

    def parse(self):
        with open(self.filename, 'r') as file:
            lines = file.read()
        
        data = []
        for line in (lines.strip().split('\n')[1:]):
            d = line.strip().split(',')
            data.append([int(i) for i in d[1:-1]])
            data[-1].append(int(d[-1]))
        
        return data
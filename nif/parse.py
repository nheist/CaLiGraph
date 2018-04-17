import os
import bz2
from rdflib import Graph


class NIFParser:

    def parse(self):
        path_to_script = os.path.dirname(os.path.realpath(__file__))
        path_to_nif_context_file = os.path.join(path_to_script, '..\\data\\nif_context_en.ttl.bz2')

        with bz2.open(path_to_nif_context_file, mode='r') as nif_context_file:
            nif_line = nif_context_file.readline()
            nif_line = nif_context_file.readline()
            g = Graph()
            g.parse(data=nif_line, format='nt')
            for stmt in g:
                print(stmt)
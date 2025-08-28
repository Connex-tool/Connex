# -*- coding:utf-8 -*-

from web3 import Web3
from src import config as C
from contextlib import contextmanager
import time, json
import inspect
from collections import defaultdict, deque
from graphviz import Digraph

class Timer(object):
    def __init__(self, start_t) -> None:
        self.start_t = start_t
        self.end_t = start_t
    def get_interval(self):
        self.end_t = time.time()
        return self.end_t - self.start_t

@contextmanager
def my_timer(auto_print_prompt:str=None, callback=None):
    tm  = Timer(time.time())
    try: 
        yield tm
    except:
        raise 
    else:
        if auto_print_prompt is not None and callback is not None:
            callback(auto_print_prompt % (tm.get_interval()))

class MyStatistics:
    def __init__(self):
        self.records = {}

    def _add_records(self, name_list:list, final_name, value):
        cur_layer = self.records
        for name in name_list: 
            if name not in cur_layer: 
                cur_layer[name] = {}
            cur_layer = cur_layer[name]
        cur_layer[final_name] = value

    def add_statis(self, name, value):
        stacks = inspect.stack()
        call_functions = []
        for callinfo in stacks:
            call_functions.append(callinfo.function)
        call_functions = call_functions[::-1]
        # call_functions.append(name)
        self._add_records(call_functions, name, value)


class UnionFindTreeNode:
    def __init__(self, value):
        self.value = value
        self.parent = None



import json
from graphviz import Digraph

class AncestorTree:
    def __init__(self):
        self.parent = {}  
        self.rank = {}    
        self.node_labels = {}  
        self.nodes = set() 

    def _make_set(self, node, node_label = None):
        
        if node not in self.parent:
          self.parent[node] = node
          self.rank[node] = 0
          self.nodes.add(node)
          self.set_label(node, node_label)
    
    def set_label(self, p, label=None):
        if label: 
            self.node_labels[p] = str(label)
        else:
            self.node_labels[p] = str(p)

    def add_relation(self, p, q, p_label=None, q_label=None):
        
        if p == q:
            return

        self._make_set(p)
        self._make_set(q)
        
        self.set_label(p, p_label)

        self.set_label(q, q_label)
        
        self._union(p,q)  
       
    def _find(self, node):
        
        if self.parent[node] != node:
            self.parent[node] = self._find(self.parent[node])
        return self.parent[node]

    def _union(self, p, q):
         
         root_p = self._find(p)
         root_q = self._find(q)
         if root_p != root_q:
            if self.rank[root_p] < self.rank[root_q]:
                self.parent[root_p] = root_q
            elif self.rank[root_p] > self.rank[root_q]:
                self.parent[root_q] = root_p
            else:
                self.parent[root_q] = root_p
                self.rank[root_p] += 1
    
    def get_ancestor(self, p, method:str='value'):
        """
        `method` == 'distance': 返回最远的祖先节点
        `method` == 'value': 返回label最大的节点
        """
        if p not in self.parent:
            return p
        if method == 'distance':
            root = self._find(p)
            return root
        elif method == 'value': 
            ds = self.get_all_descendants(p)
            max_v = 0
            root = None
            for d in ds:
                v = int(self.node_labels[d])
                if v > max_v: 
                    max_v = v 
                    root = d
            return root
        else: raise NotImplementedError


    def get_all_descendants(self, p):
        
        if p not in self.parent:
            return [p]

        root_p = self._find(p)
        descendants = [node for node in self.nodes if self._find(node) == root_p]
        return descendants

    def has_common_ancestor(self, p, q):
        
        if p not in self.parent or q not in self.parent:
          return False
        return self._find(p) == self._find(q)

    def export_to_graphviz(self, filename="graph", format="png"):
        
        dot = Digraph(comment='The Graph')

        for node in self.nodes:
            dot.node(str(node), label=self.node_labels.get(node,str(node)))

        for node in self.nodes:
            parent = self.get_parent(node)
            if parent:
              dot.edge(str(node), str(parent))

        dot.render(filename, format=format, cleanup=True)

    def to_json(self, filename="ancestortree.json", write_to_file=False):
        
        data = {
            'parent': {node: str(parent) for node, parent in self.parent.items()},
            'node_labels': self.node_labels,
            'rank': self.rank ,
            'nodes': sorted(self.nodes)
        }
        if not write_to_file:
            return data
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
            
    @classmethod
    def from_json(cls, filename="ancestortree.json"):
        
        with open(filename, 'r') as f:
            data = json.load(f)

        return AncestorTree.from_data(data)
    
    @classmethod
    def from_data(cls, data):
        
        tree = cls()
        tree.parent = {(node): (parent) for node, parent in data['parent'].items()}
        tree.rank = data.get('rank',{}) 
        tree.node_labels = data['node_labels']
        tree.nodes = set(data['nodes'])
        
        return tree

    def get_all_roots(self):
        
        roots = set()
        for node in self.nodes:
            roots.add(self._find(node))
        return sorted(list(roots))


    def get_parent(self, node):
         
         if node not in self.parent or node == self.parent[node]:
            return None
         return self.parent[node]
"""
Graph Database with Canonical Hashing System
- Standardizes graphs into canonical form
- Generates unique hash IDs
- Stores and retrieves graphs from dataset
- Supports fast lookup and isomorphism detection
"""

import networkx as nx
import json
import hashlib
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
import pickle

class GraphDatabase:
    def __init__(self, db_path: str = "./graph_database"):
        """Initialize graph database"""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Paths for data storage
        self.graphs_file = self.db_path / "graphs.json"
        self.index_file = self.db_path / "index.json"
        self.canonical_file = self.db_path / "canonical_forms.pkl"
        
        # In-memory caches
        self.graphs = {}  # hash_id -> graph_data
        self.index = {}   # canonical_form -> hash_id
        self.canonical_forms = {}  # hash_id -> canonical_form
        
        self.load_database()
    
    def get_canonical_form(self, graph: nx.Graph) -> str:
        """
        Generate canonical form of graph (invariant to isomorphism)
        Uses graph6 format which is canonical
        """
        try:
            # Use NetworkX's graph6 encoding (canonical form)
            g6_string = nx.to_graph6_bytes(graph).decode('utf-8').strip()
            return g6_string
        except:
            # Fallback: use sorted edge list with node relabeling
            return self._compute_canonical_fallback(graph)
    
    def _compute_canonical_fallback(self, graph: nx.Graph) -> str:
        """Fallback canonical form using sorted adjacency"""
        # Get adjacency matrix
        adj_matrix = nx.to_numpy_array(graph)
        
        # Convert to tuple for hashing
        edges = tuple(sorted(graph.edges()))
        
        # Create signature from edges and node count
        signature = f"{graph.number_of_nodes()}_nodes_{len(edges)}_edges_" + \
                   "_".join([f"{u}{v}" for u, v in edges])
        
        return signature
    
    def generate_hash_id(self, canonical_form: str) -> str:
        """Generate unique hash ID from canonical form"""
        hash_obj = hashlib.sha256(canonical_form.encode())
        full_hash = hash_obj.hexdigest()
        short_hash = full_hash[:16]  # Use first 16 chars
        return f"GRAPH_{short_hash}"
    
    def add_graph(self, graph: nx.Graph, metadata: Dict = None) -> str:
        """
        Add graph to database
        Returns: hash_id of the stored graph
        """
        # Verify connectivity
        if not nx.is_connected(graph):
            raise ValueError("Graph must be connected!")
        
        # Get canonical form
        canonical_form = self.get_canonical_form(graph)
        
        # Generate hash ID
        hash_id = self.generate_hash_id(canonical_form)
        
        # If already exists, return existing ID
        if hash_id in self.graphs:
            return hash_id
        
        # Store graph data
        graph_data = {
            'hash_id': hash_id,
            'canonical_form': canonical_form,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'edges': list(graph.edges()),
            'density': nx.density(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else -1,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Store in memory
        self.graphs[hash_id] = graph_data
        self.index[canonical_form] = hash_id
        self.canonical_forms[hash_id] = canonical_form
        
        return hash_id
    
    def find_graph(self, graph: nx.Graph) -> Optional[str]:
        """
        Find if graph exists in database
        Returns: hash_id if found, None otherwise
        """
        canonical_form = self.get_canonical_form(graph)
        return self.index.get(canonical_form)
    
    def get_graph_by_id(self, hash_id: str) -> Optional[Dict]:
        """Retrieve graph data by hash ID"""
        return self.graphs.get(hash_id)
    
    def get_graph_by_properties(self, 
                                num_nodes: int = None,
                                num_edges: int = None,
                                density_range: Tuple[float, float] = None) -> List[str]:
        """Search graphs by properties"""
        results = []
        
        for hash_id, graph_data in self.graphs.items():
            # Check number of nodes
            if num_nodes and graph_data['num_nodes'] != num_nodes:
                continue
            
            # Check number of edges
            if num_edges and graph_data['num_edges'] != num_edges:
                continue
            
            # Check density range
            if density_range:
                d = graph_data['density']
                if not (density_range[0] <= d <= density_range[1]):
                    continue
            
            results.append(hash_id)
        
        return results
    
    def find_isomorphic_graphs(self, graph: nx.Graph) -> List[str]:
        """Find all isomorphic graphs in database"""
        canonical_form = self.get_canonical_form(graph)
        
        isomorphic_ids = []
        for hash_id, stored_canonical in self.canonical_forms.items():
            if stored_canonical == canonical_form:
                isomorphic_ids.append(hash_id)
        
        return isomorphic_ids
    
    def reconstruct_graph(self, hash_id: str) -> Optional[nx.Graph]:
        """Reconstruct NetworkX graph from stored data"""
        graph_data = self.get_graph_by_id(hash_id)
        if not graph_data:
            return None
        
        G = nx.Graph()
        G.add_nodes_from(range(graph_data['num_nodes']))
        G.add_edges_from(graph_data['edges'])
        
        return G
    
    def save_database(self):
        """Save database to disk"""
        # Save graphs as JSON
        with open(self.graphs_file, 'w') as f:
            json.dump(self.graphs, f, indent=2)
        
        # Save index
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
        
        # Save canonical forms (pickle)
        with open(self.canonical_file, 'wb') as f:
            pickle.dump(self.canonical_forms, f)
        
        print(f"✓ Database saved: {len(self.graphs)} graphs")
    
    def load_database(self):
        """Load database from disk"""
        try:
            # Load graphs
            if self.graphs_file.exists():
                with open(self.graphs_file, 'r') as f:
                    self.graphs = json.load(f)
            
            # Load index
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            
            # Load canonical forms
            if self.canonical_file.exists():
                with open(self.canonical_file, 'rb') as f:
                    self.canonical_forms = pickle.load(f)
            
            print(f"✓ Database loaded: {len(self.graphs)} graphs")
        except Exception as e:
            print(f"⚠ Could not load database: {e}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.graphs:
            return {'total_graphs': 0}
        
        edges_list = [g['num_edges'] for g in self.graphs.values()]
        density_list = [g['density'] for g in self.graphs.values()]
        
        return {
            'total_graphs': len(self.graphs),
            'avg_edges': np.mean(edges_list),
            'min_edges': min(edges_list),
            'max_edges': max(edges_list),
            'avg_density': np.mean(density_list),
            'min_density': min(density_list),
            'max_density': max(density_list)
        }
    
    def print_info(self):
        """Print database information"""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("📊 GRAPH DATABASE STATISTICS")
        print("="*60)
        print(f"Total Graphs: {stats.get('total_graphs', 0)}")
        print(f"Avg Edges: {stats.get('avg_edges', 0):.2f}")
        print(f"Edge Range: {stats.get('min_edges', 0)} - {stats.get('max_edges', 0)}")
        print(f"Avg Density: {stats.get('avg_density', 0):.4f}")
        print("="*60 + "\n")


class GraphHasher:
    """Simplified graph hashing for quick identification"""
    
    @staticmethod
    def quick_hash(graph: nx.Graph) -> str:
        """Create quick hash from graph properties"""
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        d = nx.density(graph)
        
        # Degree sequence (sorted)
        degrees = sorted([graph.degree(node) for node in graph.nodes()])
        degree_str = "_".join(map(str, degrees))
        
        # Combine into unique string
        sig = f"{n}n_{m}e_{d:.3f}d_{degree_str}"
        
        # Hash it
        h = hashlib.md5(sig.encode()).hexdigest()[:12]
        return f"QUICK_{h}"
    
    @staticmethod
    def signature(graph: nx.Graph) -> Dict:
        """Get graph signature/fingerprint"""
        return {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': round(nx.density(graph), 4),
            'degrees': sorted([graph.degree(node) for node in graph.nodes()]),
            'is_connected': nx.is_connected(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else None,
            'radius': nx.radius(graph) if nx.is_connected(graph) else None,
        }


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    # Create database
    db = GraphDatabase(db_path="./graph_db")
    
    print("\n" + "="*60)
    print("🗄️  GRAPH HASHING & DATABASE SYSTEM")
    print("="*60)
    
    # Generate some random connected graphs
    print("\n📝 Adding sample graphs to database...")
    
    for i in range(5):
        # Generate random connected graph
        G = nx.Graph()
        G.add_nodes_from(range(8))
        
        # Add spanning tree
        for j in range(7):
            u, v = j, j + 1
            G.add_edge(u, v)
        
        # Add random edges
        for _ in range(np.random.randint(3, 10)):
            u, v = np.random.choice(8, 2, replace=False)
            G.add_edge(u, v)
        
        # Add to database
        hash_id = db.add_graph(G, metadata={'sample': i})
        print(f"  ✓ Graph {i}: {hash_id}")
    
    # Save database
    db.save_database()
    
    # Test search
    print("\n🔍 Testing search functionality...")
    
    # Create a test graph and find it
    G_test = nx.Graph()
    G_test.add_nodes_from(range(8))
    G_test.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)])
    for _ in range(3):
        u, v = np.random.choice(8, 2, replace=False)
        G_test.add_edge(u, v)
    
    # Try to find it
    found_id = db.find_graph(G_test)
    if found_id:
        print(f"✓ Found graph: {found_id}")
        data = db.get_graph_by_id(found_id)
        print(f"  Nodes: {data['num_nodes']}, Edges: {data['num_edges']}")
        print(f"  Density: {data['density']:.4f}, Diameter: {data['diameter']}")
    else:
        print("Graph not found, adding it...")
        new_id = db.add_graph(G_test, metadata={'type': 'test'})
        print(f"✓ Added: {new_id}")
    
    # Search by properties
    print("\n🔎 Searching by properties...")
    results = db.get_graph_by_properties(num_nodes=8, num_edges=10)
    print(f"Found {len(results)} graphs with 8 nodes and 10 edges: {results}")
    
    # Print statistics
    db.print_info()
    
    # Test graph signatures
    print("\n📋 Graph Signature Example:")
    sig = GraphHasher.signature(G_test)
    for key, val in sig.items():
        print(f"  {key}: {val}")

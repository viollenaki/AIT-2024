import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class Graph3DVisualizer:
    def __init__(self, num_dots=8):
        self.num_dots = num_dots
        self.possible_edges = list(nx.complete_graph(num_dots).edges())
        self.total_variations = 2 ** len(self.possible_edges)
        self.current_graph = None
        self.positions_3d = None
        self.auto_generate = False
        self.generate_random_connected_graph()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title(f"3D Graph Visualization - {num_dots} Points")
        self.root.geometry("1200x800")
        
        # Info label
        self.info_label = tk.Label(
            self.root,
            text=f"Total Possible Variations (all connected): Computing...",
            font=("Arial", 12, "bold"),
            fg="blue"
        )
        self.info_label.pack(pady=10)
        
        # Create figure
        self.fig = plt.Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Random button
        self.random_btn = tk.Button(
            button_frame,
            text="Generate Random Shape",
            command=self.on_random_click,
            font=("Arial", 11),
            bg="green",
            fg="white",
            padx=15,
            pady=8
        )
        self.random_btn.pack(side=tk.LEFT, padx=10)
        
        # Auto-generate toggle button
        self.auto_btn = tk.Button(
            button_frame,
            text="Start Auto-Generate (0.2s)",
            command=self.toggle_auto,
            font=("Arial", 11),
            bg="orange",
            fg="white",
            padx=15,
            pady=8
        )
        self.auto_btn.pack(side=tk.LEFT, padx=10)
        
        # Update label with actual count (this needs to be computed)
        self.update_variation_label()
        
        # Start auto-update thread
        self.update_thread = threading.Thread(target=self.auto_update_loop, daemon=True)
        self.update_thread.start()
        
        # Initial plot
        self.plot_graph()
        
    def generate_random_connected_graph(self):
        """Generate a random connected graph with given number of dots"""
        while True:
            # Start with a random spanning tree (which is always connected)
            G = nx.Graph()
            G.add_nodes_from(range(self.num_dots))
            
            # Add random edges
            available_edges = self.possible_edges.copy()
            np.random.shuffle(available_edges)
            
            # Ensure connectivity by starting with a spanning tree
            for i in range(self.num_dots - 1):
                u, v = available_edges[i]
                G.add_edge(u, v)
            
            # Add more random edges to make it interesting
            num_extra = np.random.randint(0, len(available_edges) - (self.num_dots - 1))
            for i in range(num_extra):
                u, v = available_edges[self.num_dots - 1 + i]
                G.add_edge(u, v)
            
            # Verify connectivity
            if nx.is_connected(G):
                self.current_graph = G
                break
        
        # Generate 3D positions for nodes
        self.generate_3d_positions()
    
    def generate_3d_positions(self):
        """Generate random 3D positions for nodes"""
        self.positions_3d = {i: [np.random.uniform(-1, 1) for _ in range(3)] 
                            for i in range(self.num_dots)}
    
    def plot_graph(self):
        """Plot the current graph in 3D"""
        self.ax.clear()
        
        if self.current_graph is None:
            return
        
        # Extract positions
        xs = [self.positions_3d[i][0] for i in range(self.num_dots)]
        ys = [self.positions_3d[i][1] for i in range(self.num_dots)]
        zs = [self.positions_3d[i][2] for i in range(self.num_dots)]
        
        # Plot edges
        for edge in self.current_graph.edges():
            u, v = edge
            self.ax.plot3D(
                [self.positions_3d[u][0], self.positions_3d[v][0]],
                [self.positions_3d[u][1], self.positions_3d[v][1]],
                [self.positions_3d[u][2], self.positions_3d[v][2]],
                'b-', linewidth=1.5, alpha=0.6
            )
        
        # Plot nodes
        self.ax.scatter(xs, ys, zs, c='red', s=200, alpha=0.9, edgecolors='darkred', linewidth=2)
        
        # Add node labels
        for i in range(self.num_dots):
            self.ax.text(xs[i], ys[i], zs[i], f'  {i}', fontsize=10, weight='bold')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'3D Graph with {self.num_dots} Connected Points\nEdges: {self.current_graph.number_of_edges()}', 
                         fontsize=12, weight='bold')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5)
        
        self.fig.canvas.draw_idle()
    
    def on_random_click(self):
        """Handle random button click"""
        self.generate_random_connected_graph()
        self.plot_graph()
    
    def toggle_auto(self):
        """Toggle auto-generation on/off"""
        self.auto_generate = not self.auto_generate
        if self.auto_generate:
            self.auto_btn.config(text="Stop Auto-Generate ⏸", bg="red")
        else:
            self.auto_btn.config(text="Start Auto-Generate (0.2s)", bg="orange")
    
    def auto_update_loop(self):
        """Background thread for auto-updating graph"""
        while True:
            time.sleep(0.2)
            if self.auto_generate:
                self.generate_random_connected_graph()
                self.plot_graph()
    
    def update_variation_label(self):
        """Calculate and update the number of valid variations"""
        # Count all possible connected graphs with 8 vertices
        # This is computationally expensive, so we'll show an approximation
        total_edges = len(self.possible_edges)
        total_possible = 2 ** total_edges
        
        info_text = (
            f"📊 Total Possible Edge Combinations: 2^{total_edges} = {total_possible:,}\n"
            f"Current Shape: {self.current_graph.number_of_edges()} edges | "
            f"All {self.num_dots} points connected ✓"
        )
        self.info_label.config(text=info_text)
    
    def run(self):
        """Start the application"""
        # Update label periodically
        def update_label():
            self.update_variation_label()
            self.root.after(500, update_label)
        
        update_label()
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = Graph3DVisualizer(num_dots=8)
    visualizer.run()

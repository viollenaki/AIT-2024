"""
Advanced 3D Graph Visualization with Animation
Shows 8 connected points in 3D space with random configurations
Auto-updates every 0.2 seconds
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from datetime import datetime

class Advanced3DGraphVisualizer:
    def __init__(self, num_dots=8):
        self.num_dots = num_dots
        self.possible_edges = list(nx.complete_graph(num_dots).edges())
        self.total_possible = 2 ** len(self.possible_edges)
        self.current_graph = None
        self.positions_3d = None
        self.auto_generate = False
        self.generation_count = 0
        self.rotation_angle = 0
        
        # Generate first graph
        self.generate_random_connected_graph()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"3D Connected Graph Visualizer - {num_dots} Points")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        # ===== TOP INFO PANEL =====
        self.create_info_panel()
        
        # ===== CANVAS AREA =====
        canvas_frame = tk.Frame(self.root, bg="white")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig = plt.Figure(figsize=(11, 7), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # ===== BUTTON PANEL =====
        self.create_button_panel()
        
        # ===== STATUS BAR =====
        self.status_bar = tk.Label(
            self.root,
            text="Ready | Click buttons to control visualization",
            font=("Arial", 10),
            bg="#e0e0e0",
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Initial plot
        self.plot_graph()
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self.auto_update_loop, daemon=True)
        self.update_thread.start()
    
    def create_info_panel(self):
        """Create information panel at the top"""
        panel = tk.Frame(self.root, bg="lightblue", relief=tk.RIDGE, borderwidth=2)
        panel.pack(fill=tk.X, padx=10, pady=10)
        
        # Main title
        title = tk.Label(
            panel,
            text=f"🔗 3D Connected Graph with {self.num_dots} Points 🔗",
            font=("Arial", 14, "bold"),
            bg="lightblue",
            fg="darkblue"
        )
        title.pack(pady=5)
        
        # Info text
        self.info_label = tk.Label(
            panel,
            text="",
            font=("Arial", 11),
            bg="lightblue",
            fg="darkgreen",
            justify=tk.LEFT
        )
        self.info_label.pack(pady=5, padx=20)
        
        # Update the label
        self.update_info_label()
    
    def create_button_panel(self):
        """Create button panel"""
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=15)
        
        # Random button
        tk.Button(
            button_frame,
            text="🎲 Random Shape",
            command=self.on_random_click,
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=15,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=10)
        
        # Auto-generate toggle
        self.auto_btn = tk.Button(
            button_frame,
            text="▶️  Auto (0.2s)",
            command=self.toggle_auto,
            font=("Arial", 11, "bold"),
            bg="#FF9800",
            fg="white",
            padx=15,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        self.auto_btn.pack(side=tk.LEFT, padx=10)
        
        # Rotate button
        tk.Button(
            button_frame,
            text="🔄 Rotate",
            command=self.toggle_rotation,
            font=("Arial", 11, "bold"),
            bg="#2196F3",
            fg="white",
            padx=15,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=10)
        
        # Reset button
        tk.Button(
            button_frame,
            text="↺ Reset",
            command=self.reset_view,
            font=("Arial", 11, "bold"),
            bg="#9C27B0",
            fg="white",
            padx=15,
            pady=10,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=10)
    
    def generate_random_connected_graph(self):
        """Generate random connected graph"""
        while True:
            G = nx.Graph()
            G.add_nodes_from(range(self.num_dots))
            
            # Create spanning tree (ensures connectivity)
            available_edges = self.possible_edges.copy()
            np.random.shuffle(available_edges)
            
            for i in range(self.num_dots - 1):
                u, v = available_edges[i]
                G.add_edge(u, v)
            
            # Add random additional edges
            num_extra = np.random.randint(2, min(8, len(available_edges) - (self.num_dots - 1)))
            for i in range(num_extra):
                if self.num_dots - 1 + i < len(available_edges):
                    u, v = available_edges[self.num_dots - 1 + i]
                    G.add_edge(u, v)
            
            if nx.is_connected(G):
                self.current_graph = G
                break
        
        # Generate 3D positions
        self.generate_3d_positions()
        self.generation_count += 1
    
    def generate_3d_positions(self):
        """Generate random 3D positions"""
        np.random.seed(None)
        self.positions_3d = {
            i: [np.random.uniform(-1, 1) for _ in range(3)]
            for i in range(self.num_dots)
        }
    
    def plot_graph(self):
        """Plot the 3D graph"""
        self.ax.clear()
        
        xs = [self.positions_3d[i][0] for i in range(self.num_dots)]
        ys = [self.positions_3d[i][1] for i in range(self.num_dots)]
        zs = [self.positions_3d[i][2] for i in range(self.num_dots)]
        
        # Plot edges with gradient colors
        edge_count = 0
        for edge in self.current_graph.edges():
            u, v = edge
            color = plt.cm.viridis(edge_count / self.current_graph.number_of_edges())
            self.ax.plot3D(
                [self.positions_3d[u][0], self.positions_3d[v][0]],
                [self.positions_3d[u][1], self.positions_3d[v][1]],
                [self.positions_3d[u][2], self.positions_3d[v][2]],
                color=color,
                linewidth=2,
                alpha=0.7
            )
            edge_count += 1
        
        # Plot nodes
        self.ax.scatter(
            xs, ys, zs,
            c='#FF5722',
            s=400,
            alpha=0.95,
            edgecolors='darkred',
            linewidth=2.5,
            marker='o'
        )
        
        # Add node labels
        for i in range(self.num_dots):
            self.ax.text(
                xs[i], ys[i], zs[i],
                f'  {i}',
                fontsize=11,
                weight='bold',
                color='darkblue'
            )
        
        # Set labels and limits
        self.ax.set_xlabel('X axis', fontsize=10, weight='bold')
        self.ax.set_ylabel('Y axis', fontsize=10, weight='bold')
        self.ax.set_zlabel('Z axis', fontsize=10, weight='bold')
        
        title_text = (
            f"3D Connected Graph | {self.num_dots} Points | "
            f"{self.current_graph.number_of_edges()} Edges | "
            f"Generation #{self.generation_count}"
        )
        self.ax.set_title(title_text, fontsize=12, weight='bold', pad=20)
        
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5)
        
        # Rotation if enabled
        if self.rotation_angle != 0:
            self.ax.view_init(elev=20, azim=self.rotation_angle)
        
        self.fig.canvas.draw_idle()
        self.update_info_label()
    
    def on_random_click(self):
        """Handle random button click"""
        self.generate_random_connected_graph()
        self.plot_graph()
        self.status_bar.config(text=f"✓ Generated new shape (#{self.generation_count})")
    
    def toggle_auto(self):
        """Toggle auto-generation"""
        self.auto_generate = not self.auto_generate
        if self.auto_generate:
            self.auto_btn.config(text="⏸  Pause", bg="#f44336")
            self.status_bar.config(text="● Auto-generating every 0.2 seconds...")
        else:
            self.auto_btn.config(text="▶️  Auto (0.2s)", bg="#FF9800")
            self.status_bar.config(text="◉ Auto-generate paused")
    
    def toggle_rotation(self):
        """Toggle view rotation"""
        if self.rotation_angle == 0:
            self.status_bar.config(text="Rotating view...")
            for angle in range(0, 360, 5):
                self.rotation_angle = angle
                self.plot_graph()
                self.root.update()
                time.sleep(0.01)
            self.rotation_angle = 0
            self.plot_graph()
        self.status_bar.config(text="Rotation completed")
    
    def reset_view(self):
        """Reset view to default"""
        self.rotation_angle = 0
        self.plot_graph()
        self.status_bar.config(text="✓ View reset to default")
    
    def auto_update_loop(self):
        """Background thread for auto-updates"""
        while True:
            time.sleep(0.2)
            if self.auto_generate:
                self.generate_random_connected_graph()
                self.plot_graph()
    
    def update_info_label(self):
        """Update information label"""
        edges = self.current_graph.number_of_edges()
        info_text = (
            f"📊 Total Possible Combinations: 2^28 = {self.total_possible:,} "
            f"| Current Edges: {edges}/{len(self.possible_edges)} "
            f"| Connected: ✓ YES"
        )
        self.info_label.config(text=info_text)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = Advanced3DGraphVisualizer(num_dots=8)
    app.run()

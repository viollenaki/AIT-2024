"""
Advanced 3D Graph Visualization with Database Integration
- Visualize 3D connected graphs
- Save/Store graphs with unique hash IDs
- Search and retrieve shapes from database
- Show graph fingerprints and properties
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from graph_database import GraphDatabase, GraphHasher

EXAMPLE_SHAPE_HASH_ID = "GRAPH_4b0b9d8a4e9cfeff"
EXAMPLE_SHAPE_EDGES = [
    (0, 5),
    (0, 1),
    (0, 2),
    (1, 6),
    (1, 4),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 7),
    (3, 6),
    (4, 7),
    (5, 6),
    (6, 7),
]

class Advanced3DGraphVizWithDB:
    def __init__(self, num_dots=8):
        self.num_dots = num_dots
        self.possible_edges = list(nx.complete_graph(num_dots).edges())
        self.total_possible = 2 ** len(self.possible_edges)
        self.current_graph = None
        self.positions_3d = None
        self.auto_generate = False
        self.generation_count = 0
        self.current_hash_id = None
        
        # Initialize database
        self.db = GraphDatabase(db_path="./graph_database")
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"3D Graph Viz + Database | 8 Points")
        self.root.geometry("1600x950")
        self.root.configure(bg="#f0f0f0")
        
        # Create menu bar
        self.create_menu_bar()
        
        # ===== TOP PANEL =====
        self.create_top_panel()
        
        # ===== MAIN CONTENT =====
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Visualization
        self.create_visualization_panel(main_frame)
        
        # Right: Info Panel
        self.create_info_panel(main_frame)
        
        # ===== BOTTOM BUTTON PANEL =====
        self.create_button_panel()
        
        # ===== STATUS BAR =====
        self.status_bar = tk.Label(
            self.root,
            text="Ready | Use buttons to control",
            font=("Arial", 9),
            bg="#e0e0e0",
            anchor=tk.W,
            padx=10,
            pady=3
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Load the fixed example graph after the UI exists
        self.load_example_shape(refresh_ui=False)
        
        # Initial plot
        self.plot_graph()
        
        # Start auto-update thread
        self.update_thread = threading.Thread(target=self.auto_update_loop, daemon=True)
        self.update_thread.start()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Database", menu=file_menu)
        file_menu.add_command(label="📊 Show Statistics", command=self.show_stats)
        file_menu.add_command(label="💾 Save Database", command=self.save_db)
        file_menu.add_separator()
        file_menu.add_command(label="🗂️  Search Graphs", command=self.open_search_window)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_top_panel(self):
        """Create top information panel"""
        panel = tk.Frame(self.root, bg="lightblue", relief=tk.RIDGE, borderwidth=2)
        panel.pack(fill=tk.X, padx=10, pady=10)
        
        title = tk.Label(
            panel,
            text=f"🔗 3D Graph Visualization with Database (8 Points)",
            font=("Arial", 13, "bold"),
            bg="lightblue",
            fg="darkblue"
        )
        title.pack(pady=5)
        
        self.subtitle_label = tk.Label(
            panel,
            text=f"Total Combinations: 2^28 = 268,435,456 | Example: {EXAMPLE_SHAPE_HASH_ID}",
            font=("Arial", 10),
            bg="lightblue",
            fg="darkgreen"
        )
        self.subtitle_label.pack(pady=3)
    
    def create_visualization_panel(self, parent):
        """Create 3D visualization panel"""
        viz_frame = tk.Frame(parent)
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.fig = plt.Figure(figsize=(9, 7), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_info_panel(self, parent):
        """Create right side info panel"""
        info_frame = tk.Frame(
            parent,
            bg="lightyellow",
            relief=tk.GROOVE,
            borderwidth=2,
            width=360,
            height=720,
        )
        info_frame.pack_propagate(False)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Title
        tk.Label(
            info_frame,
            text="📋 Graph Properties",
            font=("Arial", 11, "bold"),
            bg="lightyellow"
        ).pack(pady=5)
        
        # Scrollable text area
        self.info_text = tk.Text(
            info_frame,
            width=38,
            height=28,
            font=("Courier", 9),
            bg="white",
            relief=tk.SUNKEN,
            borderwidth=1
        )
        self.info_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.info_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)
    
    def create_button_panel(self):
        """Create button control panel"""
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        buttons = [
            ("📌 Example", self.load_example_shape, "#795548"),
            ("🎲 Random", self.on_random_click, "#4CAF50"),
            ("💾 Save Shape", self.save_current_graph, "#2196F3"),
            ("🔍 Find Shape", self.find_shape, "#FF9800"),
            ("▶️ Auto", self.toggle_auto, "#FF5722"),
            ("🔄 Rotate", self.toggle_rotation, "#9C27B0"),
            ("↺ Reset", self.reset_view, "#607D8B"),
        ]
        
        self.auto_btn = None
        for label, cmd, color in buttons:
            btn = tk.Button(
                button_frame,
                text=label,
                command=cmd,
                font=("Arial", 10, "bold"),
                bg=color,
                fg="white",
                padx=12,
                pady=8,
                cursor="hand2",
                relief=tk.RAISED,
                bd=2
            )
            btn.pack(side=tk.LEFT, padx=8)
            
            if "Auto" in label:
                self.auto_btn = btn
    
    def generate_random_connected_graph(self):
        """Generate random connected graph"""
        while True:
            G = nx.Graph()
            G.add_nodes_from(range(self.num_dots))
            
            available_edges = self.possible_edges.copy()
            np.random.shuffle(available_edges)
            
            # Spanning tree
            for i in range(self.num_dots - 1):
                u, v = available_edges[i]
                G.add_edge(u, v)
            
            # Random extra edges
            num_extra = np.random.randint(2, 8)
            for i in range(num_extra):
                if self.num_dots - 1 + i < len(available_edges):
                    u, v = available_edges[self.num_dots - 1 + i]
                    G.add_edge(u, v)
            
            if nx.is_connected(G):
                self.current_graph = G
                break
        
        self.generate_3d_positions()
        self.generation_count += 1
        
        # Try to find/add in database
        self.current_hash_id = self.db.find_graph(self.current_graph)
        if not self.current_hash_id:
            self.current_hash_id = self.db.add_graph(self.current_graph)

    def load_example_shape(self, refresh_ui=True):
        """Load a fixed example shape so the user can search for a known graph."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_dots))
        graph.add_edges_from(EXAMPLE_SHAPE_EDGES)

        self.current_graph = graph
        self.generate_3d_positions()
        self.generation_count += 1

        self.current_hash_id = self.db.find_graph(self.current_graph)
        if not self.current_hash_id:
            self.current_hash_id = self.db.add_graph(
                self.current_graph,
                metadata={"example": True, "example_hash_hint": EXAMPLE_SHAPE_HASH_ID},
            )

        if refresh_ui:
            self.plot_graph()
            self.status_bar.config(text=f"✓ Example loaded: {self.current_hash_id} | edges=14 | density=0.5000")
    
    def generate_3d_positions(self):
        """Generate fixed 3D positions (cube vertices)."""
        cube_vertices = [
            (-1, -1, -1),  # Node 0
            (1, -1, -1),   # Node 1
            (-1, 1, -1),   # Node 2
            (1, 1, -1),    # Node 3
            (-1, -1, 1),   # Node 4
            (1, -1, 1),    # Node 5
            (-1, 1, 1),    # Node 6
            (1, 1, 1),     # Node 7
        ]
        self.positions_3d = {
            i: list(cube_vertices[i])
            for i in range(self.num_dots)
        }
    
    def plot_graph(self):
        """Plot 3D graph"""
        self.ax.clear()
        
        xs = [self.positions_3d[i][0] for i in range(self.num_dots)]
        ys = [self.positions_3d[i][1] for i in range(self.num_dots)]
        zs = [self.positions_3d[i][2] for i in range(self.num_dots)]
        
        # Plot edges
        for idx, edge in enumerate(self.current_graph.edges()):
            u, v = edge
            color = plt.cm.viridis(idx / self.current_graph.number_of_edges())
            self.ax.plot3D(
                [self.positions_3d[u][0], self.positions_3d[v][0]],
                [self.positions_3d[u][1], self.positions_3d[v][1]],
                [self.positions_3d[u][2], self.positions_3d[v][2]],
                color=color,
                linewidth=2,
                alpha=0.7
            )
        
        # Plot nodes
        self.ax.scatter(xs, ys, zs, c='#FF5722', s=400, alpha=0.95,
                       edgecolors='darkred', linewidth=2.5, marker='o')
        
        # Labels
        for i in range(self.num_dots):
            self.ax.text(xs[i], ys[i], zs[i], f'  {i}', fontsize=11, weight='bold')
        
        # Settings
        self.ax.set_xlabel('X', fontsize=10, weight='bold')
        self.ax.set_ylabel('Y', fontsize=10, weight='bold')
        self.ax.set_zlabel('Z', fontsize=10, weight='bold')
        
        title = f"Graph #{self.generation_count} | Edges: {self.current_graph.number_of_edges()}"
        if self.current_hash_id:
            title += f" | ID: {self.current_hash_id[:12]}..."
        self.ax.set_title(title, fontsize=11, weight='bold', pad=20)
        
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5)
        
        self.fig.canvas.draw_idle()
        self.update_info_panel()
    
    def update_info_panel(self):
        """Update info text panel"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        # Get graph signature
        sig = GraphHasher.signature(self.current_graph)
        
        info_content = f"""
╔═══════════════════════════════╗
║     GRAPH PROPERTIES          ║
╚═══════════════════════════════╝

📊 Structural Properties:
  • Nodes: {sig['nodes']}
  • Edges: {sig['edges']}
  • Density: {sig['density']:.4f}
  • Connected: {'✓ YES' if sig['is_connected'] else '✗ NO'}

📐 Graph Metrics:
  • Diameter: {sig['diameter']}
  • Radius: {sig['radius']}
  • Degree Sequence:
    {self._format_degrees(sig['degrees'])}

🆔 Database Info:
  • Hash ID: {self.current_hash_id}
  • Total Graphs: {len(self.db.graphs)}
  • Stored Size: {self._get_db_size()}

🔍 Quick Hash:
  • {GraphHasher.quick_hash(self.current_graph)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generation #: {self.generation_count}
"""
        
        self.info_text.insert('1.0', info_content)
        self.info_text.config(state=tk.DISABLED)
    
    def _format_degrees(self, degrees):
        """Format degree sequence for display"""
        return " ".join(map(str, degrees))
    
    def _get_db_size(self):
        """Get database size info"""
        return f"{len(self.db.graphs)} entries"
    
    def on_random_click(self):
        """Generate new random graph"""
        self.generate_random_connected_graph()
        self.plot_graph()
        self.status_bar.config(text=f"✓ Generated graph #{self.generation_count}")
    
    def save_current_graph(self):
        """Save current graph to database"""
        try:
            hash_id = self.db.add_graph(
                self.current_graph,
                metadata={
                    'generation': self.generation_count,
                    'timestamp': simpledialog.askstring(
                        "Note",
                        "Add a note for this shape (optional):"
                    ) or ""
                }
            )
            self.current_hash_id = hash_id
            self.db.save_database()
            messagebox.showinfo(
                "Success",
                f"✓ Graph saved!\n\nHash ID: {hash_id}\n\nTotal graphs: {len(self.db.graphs)}"
            )
            self.status_bar.config(text=f"✓ Graph saved: {hash_id}")
            self.plot_graph()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save: {e}")
    
    def find_shape(self):
        """Open window to search database"""
        search_window = tk.Toplevel(self.root)
        search_window.title("🔍 Search Database")
        search_window.geometry("460x380")
        search_window.resizable(False, False)
        
        tk.Label(
            search_window,
            text="Search by Properties",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        tk.Label(search_window, text="Hash ID (optional, exact search):").pack()
        hash_entry = tk.Entry(search_window, width=42)
        hash_entry.pack(pady=5)
        
        # Number of edges
        tk.Label(search_window, text="Number of Edges (or leave empty):").pack()
        edges_entry = tk.Entry(search_window, width=10)
        edges_entry.pack(pady=5)
        
        # Density range
        tk.Label(search_window, text="Density Range (min-max, e.g. 0.2-0.8):").pack()
        density_entry = tk.Entry(search_window, width=20)
        density_entry.pack(pady=5)
        
        def do_search():
            try:
                hash_value = hash_entry.get().strip()
                if hash_value:
                    normalized = hash_value.replace(" ", "").replace("O", "0").replace("o", "0")
                    if not normalized.upper().startswith("GRAPH_"):
                        normalized = f"GRAPH_{normalized}"

                    graph_data = self.db.get_graph_by_id(normalized)
                    if not graph_data:
                        raise ValueError(f"Hash not found: {normalized}")

                    graph = self.db.reconstruct_graph(normalized)
                    if graph is None:
                        raise ValueError(f"Could not reconstruct graph for {normalized}")

                    self.current_graph = graph
                    self.current_hash_id = normalized
                    self.generate_3d_positions()
                    self.plot_graph()

                    result_text = (
                        f"Found exact graph by hash:\n\n"
                        f"Hash ID: {normalized}\n"
                        f"Nodes: {graph_data['num_nodes']}\n"
                        f"Edges: {graph_data['num_edges']}\n"
                        f"Density: {graph_data['density']:.4f}\n"
                        f"Diameter: {graph_data['diameter']}\n\n"
                        f"Edges list:\n{graph_data['edges']}"
                    )
                    messagebox.showinfo("Search Results", result_text)
                    self.status_bar.config(text=f"✓ Exact graph loaded: {normalized}")
                    return

                num_edges = int(edges_entry.get()) if edges_entry.get() else None
                
                density_range = None
                if density_entry.get():
                    parts = [part.strip() for part in density_entry.get().split('-', 1)]
                    if len(parts) != 2 or not parts[0] or not parts[1]:
                        raise ValueError("Use density format like 0.2-0.8")
                    d_min, d_max = map(float, parts)
                    density_range = (d_min, d_max)
                
                results = self.db.get_graph_by_properties(
                    num_nodes=8,
                    num_edges=num_edges,
                    density_range=density_range
                )
                
                result_text = f"Found {len(results)} graph(s):\n\n"
                for hash_id in results[:10]:  # Show first 10
                    data = self.db.get_graph_by_id(hash_id)
                    result_text += f"• {hash_id}\n  Edges: {data['num_edges']}, Density: {data['density']:.4f}\n"
                
                messagebox.showinfo("Search Results", result_text)
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not search: {e}")
        
        tk.Button(
            search_window,
            text="Search",
            command=do_search,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=8
        ).pack(pady=15)
    
    def toggle_auto(self):
        """Toggle auto-generation"""
        self.auto_generate = not self.auto_generate
        if self.auto_btn:
            if self.auto_generate:
                self.auto_btn.config(text="⏸ Pause", bg="#f44336")
            else:
                self.auto_btn.config(text="▶️ Auto", bg="#FF5722")
    
    def toggle_rotation(self):
        """Rotate 3D view"""
        for angle in range(0, 360, 5):
            self.ax.view_init(elev=20, azim=angle)
            self.fig.canvas.draw_idle()
            self.root.update()
            time.sleep(0.01)
    
    def reset_view(self):
        """Reset view"""
        self.ax.view_init(elev=20, azim=45)
        self.fig.canvas.draw_idle()
    
    def auto_update_loop(self):
        """Auto-update thread"""
        while True:
            time.sleep(0.2)
            if self.auto_generate:
                self.generate_random_connected_graph()
                self.plot_graph()
    
    def show_stats(self):
        """Show database statistics"""
        self.db.print_info()
        stats = self.db.get_statistics()
        
        msg = "📊 DATABASE STATISTICS\n\n"
        msg += f"Total Graphs: {stats.get('total_graphs', 0)}\n"
        msg += f"Avg Edges: {stats.get('avg_edges', 0):.2f}\n"
        msg += f"Edge Range: {stats.get('min_edges', 0)}-{stats.get('max_edges', 0)}\n"
        msg += f"Avg Density: {stats.get('avg_density', 0):.4f}\n"
        msg += f"Density Range: {stats.get('min_density', 0):.4f}-{stats.get('max_density', 0):.4f}"
        
        messagebox.showinfo("Statistics", msg)
    
    def save_db(self):
        """Save database"""
        try:
            self.db.save_database()
            messagebox.showinfo("Success", f"✓ Database saved!\n\nGraphs: {len(self.db.graphs)}")
            self.status_bar.config(text="✓ Database saved")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def open_search_window(self):
        """Open search window (same as find_shape)"""
        self.find_shape()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
3D Graph Visualization with Database

• 8 connected points in 3D space
• Total combinations: 2^28 = 268,435,456
• Uses canonical form hashing for graph identification
• Store, search, and retrieve graphs efficiently

Made with NetworkX, Matplotlib, and Tkinter
"""
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Start application"""
        self.root.mainloop()


if __name__ == "__main__":
    import tkinter.simpledialog
    app = Advanced3DGraphVizWithDB(num_dots=8)
    app.run()

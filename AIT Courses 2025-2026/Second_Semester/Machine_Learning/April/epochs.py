"""
Evolutionary and swarm intelligence optimization on continuous vectors.

- GeneticAlgorithm: selection, crossover, mutation, elitism.
- ParticleSwarmOptimizer (PSO): particles move via inertia + cognitive + social terms,
  sharing the swarm's global best — classic swarm intelligence for continuous search.

Individuals / particles use 1D numpy arrays. Fitness: higher is better when maximize=True.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FitnessCallable = Callable[[np.ndarray], float]


class SelectionMethod(Enum):
    ROULETTE = auto()
    TOURNAMENT = auto()


class CrossoverMethod(Enum):
    AVERAGE = auto()
    ONE_POINT = auto()


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """Genotype as a float vector; fitness set by the GA after evaluation."""

    genes: np.ndarray
    fitness: Optional[float] = None

    def copy(self) -> "Individual":
        return Individual(genes=self.genes.copy(), fitness=self.fitness)


# ---------------------------------------------------------------------------
# Fitness (extensibility hook)
# ---------------------------------------------------------------------------


class FitnessFunction(ABC):
    """Subclass and implement evaluate() for new problems."""

    @abstractmethod
    def evaluate(self, genes: np.ndarray) -> float:
        """Return fitness (higher = better unless maximize=False in GA)."""
        ...


class CallableFitness(FitnessFunction):
    """Wrap a plain function for quick experiments."""

    def __init__(self, fn: FitnessCallable):
        self._fn = fn

    def evaluate(self, genes: np.ndarray) -> float:
        return float(self._fn(genes))


# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------


def init_population(
    pop_size: int,
    gene_dim: int,
    low: float | np.ndarray,
    high: float | np.ndarray,
    rng: np.random.Generator,
) -> List[Individual]:
    """Uniform random initialization in [low, high] per dimension."""
    low_arr = np.broadcast_to(np.asarray(low, dtype=np.float64), (gene_dim,))
    high_arr = np.broadcast_to(np.asarray(high, dtype=np.float64), (gene_dim,))
    return [
        Individual(genes=rng.uniform(low_arr, high_arr)) for _ in range(pop_size)
    ]


def evaluate_population(
    population: List[Individual], fitness: FitnessFunction
) -> None:
    for ind in population:
        ind.fitness = fitness.evaluate(ind.genes)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def roulette_select(
    population: List[Individual],
    rng: np.random.Generator,
    maximize: bool = True,
) -> Individual:
    fits = np.array([ind.fitness for ind in population], dtype=np.float64)
    if not maximize:
        fits = -fits
    fits = fits - np.min(fits)
    total = np.sum(fits)
    if total <= 0 or not np.isfinite(total):
        return population[int(rng.integers(0, len(population)))]
    probs = fits / total
    idx = int(rng.choice(len(population), p=probs))
    return population[idx]


def tournament_select(
    population: List[Individual],
    rng: np.random.Generator,
    k: int = 3,
    maximize: bool = True,
) -> Individual:
    idxs = rng.choice(len(population), size=k, replace=False)
    candidates = [population[i] for i in idxs]
    if maximize:
        return max(candidates, key=lambda x: x.fitness)
    return min(candidates, key=lambda x: x.fitness)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover_average(parent_a: Individual, parent_b: Individual) -> Tuple[np.ndarray, np.ndarray]:
    g1, g2 = parent_a.genes.copy(), parent_b.genes.copy()
    child1 = 0.5 * (g1 + g2)
    child2 = 0.5 * (g1 + g2)  # identical; second child often duplicated in simple GAs
    # Slight diversity: blend with small noise on second child
    noise = 0.01 * (g1 - g2)
    child2 = child2 + noise
    return child1, child2


def crossover_one_point(
    parent_a: Individual,
    parent_b: Individual,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(parent_a.genes)
    if n == 1:
        return crossover_average(parent_a, parent_b)
    point = int(rng.integers(1, n))
    g1, g2 = parent_a.genes.copy(), parent_b.genes.copy()
    child1 = np.concatenate([g1[:point], g2[point:]])
    child2 = np.concatenate([g2[:point], g1[point:]])
    return child1, child2


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def mutate(
    genes: np.ndarray,
    mutation_rate: float,
    sigma: float,
    low: Optional[np.ndarray],
    high: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian noise on each gene with probability mutation_rate; clip to bounds if given."""
    out = genes.copy()
    mask = rng.random(out.shape) < mutation_rate
    n_mut = int(np.sum(mask))
    if n_mut > 0:
        out[mask] += rng.normal(0.0, sigma, size=n_mut)
    if low is not None and high is not None:
        out = np.clip(out, low, high)
    return out


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    pop_size: int = 50
    gene_dim: int = 5
    epochs: int = 100
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1
    elitism_count: int = 2
    selection: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_k: int = 3
    crossover: CrossoverMethod = CrossoverMethod.ONE_POINT
    maximize: bool = True
    seed: Optional[int] = None
    bounds_low: float | np.ndarray = -5.0
    bounds_high: float | np.ndarray = 5.0


@dataclass
class PSOConfig:
    """
    Particle Swarm Optimization (Kennedy & Eberhart).

    Velocity update (per particle, per dimension)::
        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x <- clip(x + v, bounds)

    Defaults follow a common Clerc-style choice (w ≈ 0.7298, c1 ≈ c2 ≈ 1.49618).
    """

    swarm_size: int = 50
    gene_dim: int = 5
    epochs: int = 100
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    """If None, vmax per dimension = vmax_fraction * (high - low)."""
    vmax: Optional[float] = None
    vmax_fraction: float = 0.2
    maximize: bool = True
    seed: Optional[int] = None
    bounds_low: float | np.ndarray = -5.0
    bounds_high: float | np.ndarray = 5.0


LiveVizConfig = GAConfig | PSOConfig


@dataclass
class GARunHistory:
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    best_individual_per_gen: List[Individual] = field(default_factory=list)


GenerationCallback = Callable[[int, List[Individual], GARunHistory], None]


class RealtimeGAVisualizer:
    """
    Matplotlib live view: fitness curves update each generation; optional 2D population
    scatter when gene_dim == 2. Uses interactive mode (plt.ion) + pause/flush for GUI.
    """

    def __init__(
        self,
        config: LiveVizConfig,
        *,
        title: str = "Genetic algorithm (live)",
        pause_sec: float = 0.02,
        show_population_2d: bool = True,
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.pause_sec = max(0.0, pause_sec)
        self.config = config
        self._base_title = title
        dim = config.gene_dim
        self._low = np.broadcast_to(
            np.asarray(config.bounds_low, dtype=np.float64), (dim,)
        )
        self._high = np.broadcast_to(
            np.asarray(config.bounds_high, dtype=np.float64), (dim,)
        )

        plt.ion()
        use_2d = show_population_2d and dim == 2
        if use_2d:
            self.fig, (self.ax_fit, self.ax_pop) = plt.subplots(
                1, 2, figsize=(11, 4), num=title
            )
        else:
            self.fig, self.ax_fit = plt.subplots(figsize=(8, 4), num=title)
            self.ax_pop = None

        (self._line_best,) = self.ax_fit.plot(
            [], [], label="Best (generation)", color="C0", linewidth=2
        )
        (self._line_mean,) = self.ax_fit.plot(
            [], [], label="Mean", color="C1", alpha=0.85
        )
        (self._line_ever,) = self.ax_fit.plot(
            [],
            [],
            label="Best so far",
            color="C2",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
        )
        self.ax_fit.set_xlabel("Generation")
        self.ax_fit.set_ylabel("Fitness")
        self.ax_fit.set_title(self._base_title)
        self.ax_fit.legend(loc="best")
        self.ax_fit.grid(True, alpha=0.3)

        self._scatter = None
        if self.ax_pop is not None:
            self.ax_pop.set_xlabel("Gene 0")
            self.ax_pop.set_ylabel("Gene 1")
            self.ax_pop.set_title("Population (color = fitness)")
            self.ax_pop.set_xlim(float(self._low[0]), float(self._high[0]))
            self.ax_pop.set_ylim(float(self._low[1]), float(self._high[1]))
            self.ax_pop.set_aspect("equal", adjustable="box")
            self._cbar = None

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.show()

    def __call__(
        self,
        generation: int,
        population: List[Individual],
        history: GARunHistory,
    ) -> None:
        """Callback signature matches GeneticAlgorithm.run(on_generation=...)."""
        self.update(generation, population, history)

    def update(
        self,
        generation: int,
        population: List[Individual],
        history: GARunHistory,
    ) -> None:
        gens = np.arange(1, len(history.best_fitness) + 1, dtype=np.float64)
        self._line_best.set_data(gens, history.best_fitness)
        self._line_mean.set_data(gens, history.mean_fitness)
        b = np.asarray(history.best_fitness, dtype=np.float64)
        if self.config.maximize:
            ever = np.maximum.accumulate(b)
        else:
            ever = np.minimum.accumulate(b)
        self._line_ever.set_data(gens, ever)
        self.ax_fit.relim()
        self.ax_fit.autoscale_view()

        if self.ax_pop is not None:
            xy = np.array([ind.genes[:2] for ind in population], dtype=np.float64)
            fits = np.array(
                [float(ind.fitness) if ind.fitness is not None else np.nan for ind in population],
                dtype=np.float64,
            )
            if self._scatter is None:
                self._scatter = self.ax_pop.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=fits,
                    cmap="viridis",
                    s=28,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.2,
                )
                self._cbar = self.fig.colorbar(self._scatter, ax=self.ax_pop, fraction=0.046, pad=0.04)
                self._cbar.set_label("Fitness")
            else:
                self._scatter.set_offsets(xy)
                self._scatter.set_array(fits)
                if np.all(np.isfinite(fits)) and fits.size > 0:
                    self._scatter.set_clim(float(np.min(fits)), float(np.max(fits)))

        self.ax_fit.set_title(
            f"{self._base_title} — gen {generation + 1}/{self.config.epochs}"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(self.pause_sec)

    def block_until_closed(self) -> None:
        """Keep window open after the run (interactive mode)."""
        self._plt.ioff()
        self._plt.show(block=True)

    def close(self) -> None:
        self._plt.close(self.fig)


class GeneticAlgorithm:
    """
    Standard generational GA: evaluate -> elitism -> select -> crossover -> mutate.
    """

    def __init__(
        self,
        fitness: FitnessFunction | FitnessCallable,
        config: Optional[GAConfig] = None,
    ):
        self.config = config or GAConfig()
        if isinstance(fitness, FitnessFunction):
            self.fitness_fn = fitness
        else:
            self.fitness_fn = CallableFitness(fitness)
        self.rng = np.random.default_rng(self.config.seed)
        dim = self.config.gene_dim
        self._low = np.broadcast_to(
            np.asarray(self.config.bounds_low, dtype=np.float64), (dim,)
        )
        self._high = np.broadcast_to(
            np.asarray(self.config.bounds_high, dtype=np.float64), (dim,)
        )

    def _select(self, population: List[Individual]) -> Individual:
        c = self.config
        if c.selection == SelectionMethod.ROULETTE:
            return roulette_select(population, self.rng, c.maximize)
        return tournament_select(
            population, self.rng, k=c.tournament_k, maximize=c.maximize
        )

    def _crossover(
        self, pa: Individual, pb: Individual
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.crossover == CrossoverMethod.AVERAGE:
            return crossover_average(pa, pb)
        return crossover_one_point(pa, pb, self.rng)

    def run(
        self,
        on_generation: Optional[GenerationCallback] = None,
        verbose: bool = True,
    ) -> Tuple[Individual, GARunHistory]:
        c = self.config
        history = GARunHistory()
        population = init_population(
            c.pop_size, c.gene_dim, c.bounds_low, c.bounds_high, self.rng
        )
        evaluate_population(population, self.fitness_fn)
        global_best: Optional[Individual] = None

        for gen in range(c.epochs):
            fits = np.array([ind.fitness for ind in population], dtype=np.float64)
            if c.maximize:
                sorted_idx = np.argsort(fits)[::-1]
            else:
                sorted_idx = np.argsort(fits)
            elites = [
                population[int(sorted_idx[i])].copy() for i in range(c.elitism_count)
            ]

            new_pop: List[Individual] = [e.copy() for e in elites]
            while len(new_pop) < c.pop_size:
                pa = self._select(population)
                pb = self._select(population)
                g1, g2 = self._crossover(pa, pb)
                g1 = mutate(
                    g1, c.mutation_rate, c.mutation_sigma, self._low, self._high, self.rng
                )
                g2 = mutate(
                    g2, c.mutation_rate, c.mutation_sigma, self._low, self._high, self.rng
                )
                new_pop.append(Individual(genes=g1))
                if len(new_pop) < c.pop_size:
                    new_pop.append(Individual(genes=g2))

            population = new_pop[: c.pop_size]
            evaluate_population(population, self.fitness_fn)

            fits = np.array([ind.fitness for ind in population], dtype=np.float64)
            best_idx = int(np.argmax(fits)) if c.maximize else int(np.argmin(fits))
            best_ind = population[best_idx].copy()
            assert best_ind.fitness is not None
            best_f = float(best_ind.fitness)
            mean_f = float(np.mean(fits))

            history.best_fitness.append(best_f)
            history.mean_fitness.append(mean_f)
            history.best_individual_per_gen.append(best_ind.copy())

            if global_best is None:
                global_best = best_ind.copy()
            else:
                assert global_best.fitness is not None
                if c.maximize and best_f > global_best.fitness:
                    global_best = best_ind.copy()
                elif not c.maximize and best_f < global_best.fitness:
                    global_best = best_ind.copy()

            if verbose:
                print(
                    f"Generation {gen + 1}/{c.epochs} | "
                    f"best fitness = {best_f:.6f} | mean fitness = {mean_f:.6f}"
                )
            if on_generation is not None:
                on_generation(gen, population, history)

        assert global_best is not None
        return global_best.copy(), history


def plot_fitness_history(
    history: GARunHistory,
    title: str = "Fitness over generations",
    show: bool = True,
) -> None:
    """Optional matplotlib plot (call only if matplotlib is installed)."""
    import matplotlib.pyplot as plt

    gens = range(1, len(history.best_fitness) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(gens, history.best_fitness, label="Best fitness", linewidth=2)
    plt.plot(gens, history.mean_fitness, label="Mean fitness", alpha=0.8)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Example: sphere minimization (negate for maximization inside fitness)
# ---------------------------------------------------------------------------


def example_sphere_maximize(genes: np.ndarray) -> float:
    """Maximize negative squared distance to origin (optimum at 0)."""
    return float(-np.sum(genes**2))


def example_rastrigin_maximize(genes: np.ndarray, a: float = 10.0) -> float:
    """Classic Rastrigin; negate for maximization."""
    n = len(genes)
    val = a * n + np.sum(genes**2 - a * np.cos(2 * np.pi * genes))
    return float(-val)


if __name__ == "__main__":
    import sys

    live = "--live" in sys.argv

    if live:
        cfg = GAConfig(
            pop_size=100,
            gene_dim=2,
            epochs=120,
            mutation_rate=0.12,
            mutation_sigma=0.25,
            elitism_count=2,
            selection=SelectionMethod.TOURNAMENT,
            tournament_k=3,
            crossover=CrossoverMethod.ONE_POINT,
            maximize=True,
            seed=42,
            bounds_low=-5.12,
            bounds_high=5.12,
        )
        try:
            viz = RealtimeGAVisualizer(
                cfg,
                title="Rastrigin — live GA",
                pause_sec=0.03,
                show_population_2d=True,
            )
        except ImportError:
            print("Install matplotlib for live view: pip install matplotlib")
            raise SystemExit(1) from None
        ga = GeneticAlgorithm(example_rastrigin_maximize, cfg)
        best, hist = ga.run(on_generation=viz, verbose=False)
        print("\nBest individual genes:", best.genes)
        print("Best fitness:", best.fitness)
        viz.block_until_closed()
    else:
        cfg = GAConfig(
            pop_size=80,
            gene_dim=8,
            epochs=60,
            mutation_rate=0.15,
            mutation_sigma=0.2,
            elitism_count=2,
            selection=SelectionMethod.TOURNAMENT,
            tournament_k=3,
            crossover=CrossoverMethod.ONE_POINT,
            maximize=True,
            seed=42,
            bounds_low=-5.12,
            bounds_high=5.12,
        )
        ga = GeneticAlgorithm(example_rastrigin_maximize, cfg)
        best, hist = ga.run()
        print("\nBest individual genes:", best.genes)
        print("Best fitness:", best.fitness)

        try:
            plot_fitness_history(hist, title="Rastrigin — GA fitness trace")
        except ImportError:
            print("(matplotlib not installed; skipping plot)")

"""
Evolutionary and swarm intelligence optimization on continuous vectors.

- GeneticAlgorithm: selection, crossover, mutation, elitism.
- ParticleSwarmOptimizer (PSO): swarm of particles pulled toward personal and global bests.

Run ``python epochs.py`` for a simple blue-eye vs brown-eye toy demo with a live plot.
Run ``python epochs.py --rastrigin`` for a harder math benchmark. Higher fitness is better when maximize=True.
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
    low_bias_exponent: float = 1.0,
) -> List[Individual]:
    """
    Random initialization in [low, high] per dimension.

    If low_bias_exponent > 1, each gene is ``u ** exponent`` with u ~ Uniform(0,1),
    then scaled to the box — values cluster toward *low* (more blue eyes in the demo).
    """
    low_arr = np.broadcast_to(np.asarray(low, dtype=np.float64), (gene_dim,))
    high_arr = np.broadcast_to(np.asarray(high, dtype=np.float64), (gene_dim,))
    span = high_arr - low_arr
    if float(low_bias_exponent) == 1.0:
        return [
            Individual(genes=rng.uniform(low_arr, high_arr)) for _ in range(pop_size)
        ]
    exp = float(low_bias_exponent)
    return [
        Individual(
            genes=low_arr + (rng.uniform(0.0, 1.0, size=gene_dim) ** exp) * span
        )
        for _ in range(pop_size)
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
    # > 1 biases random start toward bounds_low (e.g. more blue-eyed starts in the demo).
    init_low_bias_exponent: float = 1.0


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
    Live matplotlib dashboard for GA or PSO: fitness vs epoch, on-plot epoch stats,
    optional 2D scatter when gene_dim == 2. Uses plt.ion() + pause/flush for GUI.
    """

    def __init__(
        self,
        config: LiveVizConfig,
        *,
        title: str = "Genetic algorithm (live)",
        pause_sec: float = 0.02,
        show_population_2d: bool = True,
        footer_text: str = "",
        pop_x_label: str = "",
        pop_y_label: str = "",
        fitness_y_label: str = "Fitness",
        scatter_cbar_label: str = "Score",
        scatter_eye_types: bool = False,
        eye_type_threshold: float = 0.5,
        include_initial_live_frame: bool = False,
    ):
        import matplotlib.pyplot as plt

        self._plt = plt
        self.pause_sec = max(0.0, pause_sec)
        self.config = config
        self._base_title = title
        self._is_pso = isinstance(config, PSOConfig)
        self._footer_text = footer_text
        self._fitness_y_label = fitness_y_label
        self._scatter_eye_types = scatter_eye_types
        self._eye_threshold = float(eye_type_threshold)
        self._include_initial_live_frame = include_initial_live_frame
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
            self.fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), num=title)
            # People scatter on the left, score curves on the right (easier to see)
            if scatter_eye_types:
                self.ax_pop, self.ax_fit = axes[0], axes[1]
            else:
                self.ax_fit, self.ax_pop = axes[0], axes[1]
        else:
            self.fig, self.ax_fit = plt.subplots(figsize=(8, 4), num=title)
            self.ax_pop = None

        best_curve_label = "Global best" if self._is_pso else "Best (generation)"
        (self._line_best,) = self.ax_fit.plot(
            [], [], label=best_curve_label, color="C0", linewidth=2
        )
        (self._line_mean,) = self.ax_fit.plot(
            [], [], label="Mean (swarm)" if self._is_pso else "Mean", color="C1", alpha=0.85
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
        self.ax_fit.set_xlabel("Epoch")
        self.ax_fit.set_ylabel(fitness_y_label)
        self.ax_fit.set_title(self._base_title)
        self.ax_fit.legend(loc="best")
        self.ax_fit.grid(True, alpha=0.3)

        self._epoch_box = self.ax_fit.text(
            0.98,
            0.02,
            "",
            transform=self.ax_fit.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#666666",
                "alpha": 0.92,
            },
            zorder=5,
        )

        self._scatter = None
        self._scatter_cbar_label = scatter_cbar_label
        self._cbar = None
        self._eye_legend_added = False
        if self.ax_pop is not None:
            self.ax_pop.set_xlabel(pop_x_label or "Gene 0")
            self.ax_pop.set_ylabel(pop_y_label or "Gene 1")
            self.ax_pop.set_title(
                "Each dot = one person (blue eyes vs brown eyes)"
                if scatter_eye_types
                else "All tries (color = score)"
            )
            if scatter_eye_types:
                span_x = float(self._high[0] - self._low[0])
                span_y = float(self._high[1] - self._low[1])
                pad = 0.15 * max(span_x, span_y, 1e-6)
                self.ax_pop.set_xlim(float(self._low[0]) - pad, float(self._high[0]) + pad)
                self.ax_pop.set_ylim(float(self._low[1]) - pad, float(self._high[1]) + pad)
                self.ax_pop.set_aspect("equal", adjustable="box")
            else:
                self.ax_pop.set_xlim(float(self._low[0]), float(self._high[0]))
                self.ax_pop.set_ylim(float(self._low[1]), float(self._high[1]))
                self.ax_pop.set_aspect("auto")

        if footer_text:
            self.fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            self.fig.text(
                0.5,
                0.02,
                footer_text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
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

        fits = np.array(
            [
                float(ind.fitness) if ind.fitness is not None else np.nan
                for ind in population
            ],
            dtype=np.float64,
        )
        valid_f = np.isfinite(fits)
        pct_brown = (
            100.0 * float(np.mean(fits[valid_f] >= self._eye_threshold))
            if self._scatter_eye_types and np.any(valid_f)
            else 0.0
        )

        if self.ax_pop is not None:
            xy = np.array([ind.genes[:2] for ind in population], dtype=np.float64)
            if self._scatter_eye_types:
                is_brown = np.zeros(len(fits), dtype=bool)
                is_brown[valid_f] = fits[valid_f] >= self._eye_threshold
                colors = np.where(is_brown, "#5C4033", "#1E5AA8").tolist()
                if self._scatter is None:
                    self._scatter = self.ax_pop.scatter(
                        xy[:, 0],
                        xy[:, 1],
                        c=colors,
                        s=32,
                        alpha=0.88,
                        edgecolors="k",
                        linewidths=0.25,
                    )
                    if not self._eye_legend_added:
                        from matplotlib.patches import Patch

                        self.ax_pop.legend(
                            handles=[
                                Patch(
                                    facecolor="#1E5AA8",
                                    edgecolor="k",
                                    label="Blue eyes",
                                ),
                                Patch(
                                    facecolor="#5C4033",
                                    edgecolor="k",
                                    label="Brown eyes",
                                ),
                            ],
                            loc="lower right",
                            fontsize=8,
                        )
                        self._eye_legend_added = True
                else:
                    self._scatter.set_offsets(xy)
                    self._scatter.set_facecolors(colors)
            else:
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
                    self._cbar = self.fig.colorbar(
                        self._scatter, ax=self.ax_pop, fraction=0.046, pad=0.04
                    )
                    self._cbar.set_label(self._scatter_cbar_label)
                else:
                    self._scatter.set_offsets(xy)
                    self._scatter.set_array(fits)
                    if np.all(np.isfinite(fits)) and fits.size > 0:
                        self._scatter.set_clim(
                            float(np.min(fits)), float(np.max(fits))
                        )

        bf = history.best_fitness[-1]
        mf = history.mean_fitness[-1]
        algo = "PSO" if self._is_pso else "GA"
        total_steps = self.config.epochs + (
            1 if self._include_initial_live_frame else 0
        )
        step = len(history.best_fitness)
        self.ax_fit.set_title(
            f"{self._base_title} — step {step}/{total_steps} ({algo})"
        )
        if self._scatter_eye_types:
            self._epoch_box.set_text(
                f"Step {step} / {total_steps}\n"
                f"Avg brown score {mf:.3f}\n"
                f"Brown eyes ~{pct_brown:.0f}%"
            )
        else:
            self._epoch_box.set_text(
                f"Step {step} / {total_steps}\nBest  {bf:.6f}\nMean  {mf:.6f}"
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
            c.pop_size,
            c.gene_dim,
            c.bounds_low,
            c.bounds_high,
            self.rng,
            c.init_low_bias_exponent,
        )
        evaluate_population(population, self.fitness_fn)
        global_best: Optional[Individual] = None

        if on_generation is not None:
            fits0 = np.array([ind.fitness for ind in population], dtype=np.float64)
            best_idx0 = int(np.argmax(fits0)) if c.maximize else int(np.argmin(fits0))
            best_ind0 = population[best_idx0].copy()
            assert best_ind0.fitness is not None
            best_f0 = float(best_ind0.fitness)
            mean_f0 = float(np.mean(fits0))
            history.best_fitness.append(best_f0)
            history.mean_fitness.append(mean_f0)
            history.best_individual_per_gen.append(best_ind0.copy())
            global_best = best_ind0.copy()
            on_generation(-1, population, history)

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


def _is_better_fitness(a: float, b: float, maximize: bool) -> bool:
    return a > b if maximize else a < b


@dataclass
class Particle:
    """PSO agent: position, velocity, and personal best (cognitive memory)."""

    position: np.ndarray
    velocity: np.ndarray
    pbest_position: np.ndarray
    pbest_fitness: float
    fitness: float = 0.0


def particles_as_individuals(particles: List[Particle]) -> List[Individual]:
    return [
        Individual(genes=p.position.copy(), fitness=float(p.fitness))
        for p in particles
    ]


def _global_best_from_pbests(
    particles: List[Particle], maximize: bool
) -> Tuple[np.ndarray, float]:
    fits = np.array([p.pbest_fitness for p in particles], dtype=np.float64)
    k = int(np.argmax(fits)) if maximize else int(np.argmin(fits))
    p = particles[k]
    return p.pbest_position.copy(), float(p.pbest_fitness)


class ParticleSwarmOptimizer:
    """
    Canonical PSO: the swarm shares a global best (social attraction); each particle
    remembers its personal best (cognitive). Velocities are clamped to stabilize search.
    """

    def __init__(
        self,
        fitness: FitnessFunction | FitnessCallable,
        config: Optional[PSOConfig] = None,
    ):
        self.config = config or PSOConfig()
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
        span = self._high - self._low
        c = self.config
        if c.vmax is not None:
            self._vmax = np.full(dim, float(c.vmax), dtype=np.float64)
        else:
            self._vmax = np.maximum(span * float(c.vmax_fraction), 1e-6)

    def _make_swarm(self) -> List[Particle]:
        c = self.config
        rng = self.rng
        particles: List[Particle] = []
        worst = float("-inf") if c.maximize else float("inf")
        for _ in range(c.swarm_size):
            x = rng.uniform(self._low, self._high)
            v = rng.uniform(-self._vmax, self._vmax)
            particles.append(
                Particle(
                    position=x,
                    velocity=v,
                    pbest_position=x.copy(),
                    pbest_fitness=worst,
                    fitness=0.0,
                )
            )
        return particles

    def run(
        self,
        on_generation: Optional[GenerationCallback] = None,
        verbose: bool = True,
    ) -> Tuple[Individual, GARunHistory]:
        c = self.config
        history = GARunHistory()
        particles = self._make_swarm()

        for p in particles:
            p.fitness = self.fitness_fn.evaluate(p.position)
            p.pbest_position = p.position.copy()
            p.pbest_fitness = float(p.fitness)

        gbest_pos, gbest_fit = _global_best_from_pbests(particles, c.maximize)
        global_best = Individual(genes=gbest_pos.copy(), fitness=gbest_fit)

        for gen in range(c.epochs):
            for p in particles:
                r1 = self.rng.random(c.gene_dim)
                r2 = self.rng.random(c.gene_dim)
                p.velocity = (
                    c.w * p.velocity
                    + c.c1 * r1 * (p.pbest_position - p.position)
                    + c.c2 * r2 * (gbest_pos - p.position)
                )
                p.velocity = np.clip(p.velocity, -self._vmax, self._vmax)
                p.position = np.clip(p.position + p.velocity, self._low, self._high)
                p.fitness = float(self.fitness_fn.evaluate(p.position))
                if _is_better_fitness(p.fitness, p.pbest_fitness, c.maximize):
                    p.pbest_fitness = p.fitness
                    p.pbest_position = p.position.copy()

            gbest_pos, gbest_fit = _global_best_from_pbests(particles, c.maximize)
            fits_curr = np.array([p.fitness for p in particles], dtype=np.float64)
            mean_f = float(np.mean(fits_curr))

            history.best_fitness.append(gbest_fit)
            history.mean_fitness.append(mean_f)
            history.best_individual_per_gen.append(
                Individual(genes=gbest_pos.copy(), fitness=gbest_fit)
            )

            assert global_best.fitness is not None
            if _is_better_fitness(gbest_fit, global_best.fitness, c.maximize):
                global_best = Individual(genes=gbest_pos.copy(), fitness=gbest_fit)

            if verbose:
                print(
                    f"PSO iteration {gen + 1}/{c.epochs} | "
                    f"global best = {gbest_fit:.6f} | mean fitness = {mean_f:.6f}"
                )
            if on_generation is not None:
                on_generation(gen, particles_as_individuals(particles), history)

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


def example_blue_brown_eye_score(genes: np.ndarray) -> float:
    """
    Toy classroom model only (not real genetics).

    Each person has two numbers from 0 to 1 (two simplified "factors").
    The score is their average. Higher score = more brown-eye tendency in this toy rule.
    If the score is at least 0.5 we draw the person as brown-eyed, else blue-eyed.
    The GA is set up to *maximize* this score so the group shifts toward more brown eyes.
    """
    a = float(np.clip(genes[0], 0.0, 1.0))
    b = float(np.clip(genes[1], 0.0, 1.0))
    return 0.5 * (a + b)


# ``python epochs.py`` → blue vs brown eye toy demo + live window.
# ``python epochs.py --rastrigin`` → math benchmark (8D GA, plot at end).
# ``python epochs.py --pso`` / ``--pso-live`` / ``--live`` → see branches below.


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    use_pso = "--pso" in argv
    use_rastrigin = "--rastrigin" in argv
    live = "--live" in argv or "--pso-live" in argv
    pso_live_only = "--pso-live" in argv
    pso_live_2d = "--pso-live-2d" in argv

    _default_pso_cfg = PSOConfig(
        swarm_size=80,
        gene_dim=8,
        epochs=60,
        w=0.7298,
        c1=1.49618,
        c2=1.49618,
        vmax_fraction=0.2,
        maximize=True,
        seed=42,
        bounds_low=-5.12,
        bounds_high=5.12,
    )

    if pso_live_2d:
        pcfg = PSOConfig(
            swarm_size=100,
            gene_dim=2,
            epochs=120,
            w=0.7298,
            c1=1.49618,
            c2=1.49618,
            vmax_fraction=0.25,
            maximize=True,
            seed=42,
            bounds_low=-5.12,
            bounds_high=5.12,
        )
        try:
            viz = RealtimeGAVisualizer(
                pcfg,
                title="Rastrigin — PSO 2D demo (scatter)",
                pause_sec=0.03,
                show_population_2d=True,
            )
        except ImportError:
            print("Install matplotlib for live view: pip install matplotlib")
            raise SystemExit(1) from None
        pso = ParticleSwarmOptimizer(example_rastrigin_maximize, pcfg)
        best, hist = pso.run(on_generation=viz, verbose=False)
        print("\nBest position (genes):", best.genes)
        print("Best fitness:", best.fitness)
        viz.block_until_closed()
    elif (use_pso and live) or pso_live_only:
        pcfg = _default_pso_cfg
        try:
            viz = RealtimeGAVisualizer(
                pcfg,
                title="Rastrigin — PSO (live epochs)",
                pause_sec=0.03,
                show_population_2d=False,
            )
        except ImportError:
            print("Install matplotlib for live view: pip install matplotlib")
            raise SystemExit(1) from None
        pso = ParticleSwarmOptimizer(example_rastrigin_maximize, pcfg)
        best, hist = pso.run(on_generation=viz, verbose=False)
        print("\nBest position (genes):", best.genes)
        print("Best fitness:", best.fitness)
        viz.block_until_closed()
    elif live:
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
                title="Rastrigin — GA (live epochs)",
                pause_sec=0.03,
                show_population_2d=True,
                include_initial_live_frame=True,
            )
        except ImportError:
            print("Install matplotlib for live view: pip install matplotlib")
            raise SystemExit(1) from None
        ga = GeneticAlgorithm(example_rastrigin_maximize, cfg)
        best, hist = ga.run(on_generation=viz, verbose=False)
        print("\nBest individual genes:", best.genes)
        print("Best fitness:", best.fitness)
        viz.block_until_closed()
    elif use_pso:
        pcfg = _default_pso_cfg
        pso = ParticleSwarmOptimizer(example_rastrigin_maximize, pcfg)
        best, hist = pso.run()
        print("\nBest position (genes):", best.genes)
        print("Best fitness:", best.fitness)
        try:
            plot_fitness_history(hist, title="Rastrigin — PSO (swarm) fitness trace")
        except ImportError:
            print("(matplotlib not installed; skipping plot)")
    elif use_rastrigin:
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
    else:
        # Default: toy "population" shifting toward more brown eyes (not real biology)
        cfg = GAConfig(
            pop_size=70,
            gene_dim=2,
            epochs=90,
            mutation_rate=0.18,
            mutation_sigma=0.12,
            elitism_count=2,
            selection=SelectionMethod.TOURNAMENT,
            tournament_k=3,
            crossover=CrossoverMethod.ONE_POINT,
            maximize=True,
            seed=42,
            bounds_low=np.array([0.0, 0.0], dtype=np.float64),
            bounds_high=np.array([1.0, 1.0], dtype=np.float64),
            init_low_bias_exponent=3.0,
        )
        footer = (
            "Simple story: each dot is one person. Blue or brown is decided by a fake score (average of two numbers).\n"
            "The program rewards higher scores, so over time you should see more brown-eyed people.\n"
            "This is only for learning — real eye color is much more complex."
        )
        print(
            "Opening a window: a toy population of people with blue or brown eyes.\n"
            "The search tries to push the group toward as many brown eyes as possible.\n"
            "(Use python epochs.py --rastrigin for the harder math benchmark.)\n"
        )
        try:
            viz = RealtimeGAVisualizer(
                cfg,
                title="Toy demo: population leaning toward brown eyes",
                pause_sec=0.035,
                show_population_2d=True,
                footer_text=footer,
                pop_x_label="Factor A (0 = low, 1 = high)",
                pop_y_label="Factor B (0 = low, 1 = high)",
                fitness_y_label="Brown-eye score (higher = more brown)",
                scatter_eye_types=True,
                eye_type_threshold=0.5,
                include_initial_live_frame=True,
            )
        except ImportError:
            print("matplotlib not found — running without a window.")
            ga = GeneticAlgorithm(example_blue_brown_eye_score, cfg)
            best, hist = ga.run(verbose=True)
        else:
            ga = GeneticAlgorithm(example_blue_brown_eye_score, cfg)
            best, hist = ga.run(on_generation=viz, verbose=False)
            viz.block_until_closed()

        print("\n--- Result ---")
        assert best.fitness is not None
        bf = float(best.fitness)
        print(
            f"Best person: factor A = {best.genes[0]:.3f}, factor B = {best.genes[1]:.3f} "
            f"(toy brown-eye score = {bf:.3f})"
        )
        print(
            "In this toy rule, that counts as brown eyes."
            if bf >= 0.5
            else "Still below the brown cutoff in the toy rule."
        )

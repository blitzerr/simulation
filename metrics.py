import simpy
import collections

# Import all our simulation classes
from simulation_components import Job, JobQueue, InstancePool


class MetricsCollector:
    """
    Collects and stores simulation metrics.
    """

    def __init__(self):
        self.queue_lengths_by_color: dict[str, list[tuple[float, int]]] = (
            collections.defaultdict(list)
        )  # {color: [(time, length), ...]}
        self.pool_utilization: list[tuple[float, float]] = (
            []
        )  # [(time, utilization), ...]
        self.job_wait_times: list[float] = []  # [wait_time, ...]
        self.job_completion_times: list[tuple[int, float]] = (
            []
        )  # [(job_id, completion_time), ...]
        self.submitted_jobs: list[Job] = []
        self.instance_demand_over_time: dict[int, int] = collections.defaultdict(int)
        self.job_start_times: list[tuple[int, float]] = (
            []
        )  # [(job_id, start_time), ...]
        self.instance_counts: list[tuple[float, int, int, int]] = (
            []
        )  # [(time, total, free, busy), ...]

    def record_queue_length(self, time: float, lengths_by_color: dict[str, int]):
        for color, length in lengths_by_color.items():
            self.queue_lengths_by_color[color].append((time, length))
        # Also record total queue length
        self.queue_lengths_by_color["total"].append(
            (time, sum(lengths_by_color.values()))
        )

    def record_pool_utilization(self, time: float, utilization: float):
        self.pool_utilization.append((time, utilization))

    def record_instance_counts(self, time: float, total: int, free: int, busy: int):
        self.instance_counts.append((time, total, free, busy))

    def record_job_start(self, job: Job, start_time: float):
        job.start_time = start_time
        self.job_start_times.append((job.id, start_time))

    def record_job_completion(self, job: Job, end_time: float):
        job.end_time = end_time
        job.wait_time = (
            job.start_time - job.submission_time if job.start_time is not None else None
        )
        if job.wait_time is not None:
            self.job_wait_times.append(job.wait_time)
        self.job_completion_times.append((job.id, end_time))

    def get_average_job_wait_time(self) -> float:
        if not self.job_wait_times:
            return 0.0
        return sum(self.job_wait_times) / len(self.job_wait_times)

    def get_average_pool_utilization(self) -> float:
        if not self.pool_utilization:
            return 0.0
        return sum([u for _, u in self.pool_utilization]) / len(self.pool_utilization)

    def get_max_queue_length_by_color(self) -> dict[str, int]:
        max_lengths: dict[str, int] = {}
        for color, data_points in self.queue_lengths_by_color.items():
            if data_points:
                max_lengths[color] = max([length for _, length in data_points])
        return max_lengths

    def get_final_metrics(self) -> dict[str, float | dict[str, int]]:
        return {
            "avg_job_wait_time": self.get_average_job_wait_time(),
            "avg_pool_utilization": self.get_average_pool_utilization(),
            "max_queue_length_by_color": self.get_max_queue_length_by_color(),
            "total_jobs_completed": len(self.job_completion_times),
        }

    def print_summary(self):
        """Prints a final summary of the simulation results."""
        print("\n--- Simulation Summary ---")
        summary = self.get_final_metrics()
        print(f"Total Jobs Completed: {summary['total_jobs_completed']}")
        print(f"Average Job Wait Time: {summary['avg_job_wait_time']:.2f}")
        print(f"Average Pool Utilization: {summary['avg_pool_utilization']:.2%}")
        print("Max Queue Lengths:")
        for color, length in summary["max_queue_length_by_color"].items():
            print(f"  - {color.capitalize()}: {length}")
        print("--------------------------")
        
    def plot(self):
        """
        Generates and displays plots from the collected simulation metrics.
        """
        import matplotlib.pyplot as plt

        print("\n--- Generating plots ---")
        fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
        fig.suptitle("Job Execution System Simulation Results", fontsize=16)

        # Plot 1: Queue Length Over Time
        ax1 = axs[0]
        for color, data in self.queue_lengths_by_color.items():
            if data:
                times, lengths = zip(*data)
                ax1.plot(times, lengths, label=f"Queue ({color})")
        ax1.set_ylabel("Queue Length")
        ax1.set_title("Job Queue Length Over Time")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Instance Pool Size & Composition
        ax2 = axs[1]
        if self.instance_counts:
            times, total, free, busy = zip(*self.instance_counts)
            ax2.plot(times, total, label="Total Instances", color="black")
            ax2.plot(times, busy, label="Busy Instances", color="red")
            ax2.plot(times, free, label="Free Instances", color="green")
            ax2.set_ylabel("Number of Instances")
            ax2.set_title("Instance Pool Size and Composition")
            ax2.legend()
            ax2.grid(True)

        # Plot 3: Pool Utilization
        ax3 = axs[2]
        if self.pool_utilization:
            times, utils = zip(*self.pool_utilization)
            ax3.plot(times, [u * 100 for u in utils], label="Utilization %")
            ax3.axhline(y=85, color="r", linestyle="--", label="High Threshold (85%)")
            ax3.axhline(y=60, color="g", linestyle="--", label="Low Threshold (60%)")
            ax3.set_ylabel("Utilization (%)")
            ax3.set_title("Instance Pool Utilization")
            ax3.set_ylim(0, 110)
            ax3.legend()
            ax3.grid(True)

        # Plot 4: Job Wait Time Distribution
        ax4 = axs[3]
        if self.job_wait_times:
            ax4.hist(self.job_wait_times, bins=30, alpha=0.75)
            avg_wait = self.get_average_job_wait_time()
            ax4.axvline(
                avg_wait,
                color="r",
                linestyle="dashed",
                linewidth=2,
                label=f"Avg Wait: {avg_wait:.2f}",
            )
            ax4.set_xlabel("Time")
            ax4.set_ylabel("Number of Jobs")
            ax4.set_title("Job Wait Time Distribution")
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


class MetricsMonitor:
    """A process that periodically records metrics from various components."""

    def __init__(
        self,
        env: simpy.Environment,
        job_queue: JobQueue,
        instance_pool: InstancePool,
        metrics: MetricsCollector,
    ):
        self.env = env
        self.job_queue = job_queue
        self.instance_pool = instance_pool
        self.metrics = metrics
        self.action = env.process(self.run())

    def run(self):
        """The main monitoring loop."""
        while True:
            # Record queue lengths by color
            self.metrics.record_queue_length(
                self.env.now, self.job_queue.get_queue_length_by_color()
            )
            # Record pool utilization
            self.metrics.record_pool_utilization(
                self.env.now, self.instance_pool.get_utilization()
            )
            # Record instance counts
            self.metrics.record_instance_counts(
                self.env.now,
                self.instance_pool.get_total_instances(),
                self.instance_pool.get_free_instances_count(),
                self.instance_pool.get_busy_instances_count(),
            )
            yield self.env.timeout(1)

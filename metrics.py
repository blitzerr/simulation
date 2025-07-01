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
        self.job_submission_times: list[float] = []
        self.job_start_times: list[tuple[int, float]] = (
            []
        )  # [(job_id, start_time), ...]
        self.instance_counts: list[tuple[float, int, int, int]] = (
            []
        )  # [(time, total, free, busy), ...]
        self.instance_demand_over_time: list[tuple[float, int]] = (
            []
        )  # [(time, instance_demand), ...]

        # --- New Metrics ---
        self.gridlock_events: list[tuple[float, bool]] = []
        self.oldest_job_age: list[tuple[float, float]] = []
        # --- End New Metrics ---

        # Internal state for tracking submissions
        self._seen_job_ids: set[int] = set()

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

    def record_instance_demand(self, time: float, demand: int):
        self.instance_demand_over_time.append((time, demand))

    def record_gridlock_event(self, time: float, is_stuck: bool):
        self.gridlock_events.append((time, is_stuck))

    def record_oldest_job_age(self, time: float, age: float):
        self.oldest_job_age.append((time, age))

    def record_job_submission(self, job: Job):
        if job.id not in self._seen_job_ids:
            self.job_submission_times.append(job.submission_time)
            self._seen_job_ids.add(job.id)

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

    def get_final_metrics(self) -> dict:
        total_submissions = len(self.job_submission_times)
        total_starts = len(self.job_start_times)
        queue_stability_ratio = (
            total_starts / total_submissions if total_submissions > 0 else 0
        )

        avg_demand = (
            sum(d for _, d in self.instance_demand_over_time)
            / len(self.instance_demand_over_time)
            if self.instance_demand_over_time
            else 0
        )
        avg_capacity = (
            sum(t for _, t, _, _ in self.instance_counts) / len(self.instance_counts)
            if self.instance_counts
            else 0
        )
        control_system_stability_ratio = (
            avg_demand / avg_capacity if avg_capacity > 0 else 0
        )

        # --- New Metric Calculation ---
        total_gridlock_time = sum(1 for _, stuck in self.gridlock_events if stuck)
        # --- End New Metric Calculation ---

        return {
            "avg_job_wait_time": self.get_average_job_wait_time(),
            "avg_pool_utilization": self.get_average_pool_utilization(),
            "max_queue_length_by_color": self.get_max_queue_length_by_color(),
            "total_jobs_completed": len(self.job_completion_times),
            "queue_stability": queue_stability_ratio,
            "control_system_stability": control_system_stability_ratio,
            "total_gridlock_time": total_gridlock_time,
        }

    def print_summary(self):
        """Prints a final summary of the simulation results."""
        print("\n--- Simulation Summary ---")
        summary = self.get_final_metrics()
        print(f"Total Jobs Completed: {summary['total_jobs_completed']}")
        print(f"Average Job Wait Time: {summary['avg_job_wait_time']:.2f}")
        print(f"Average Pool Utilization: {summary['avg_pool_utilization']:.2%}")

        print(f"Queue Stability (Starts/Submissions): {summary['queue_stability']:.2%}")
        print(
            "Control System Stability (Avg Demand/Avg Capacity): "
            f"{summary['control_system_stability']:.2%}"
        )
        # --- New Summary Output ---
        print(f"Total Time in Gridlock: {summary['total_gridlock_time']} steps")
        # --- End New Summary Output ---

        print("Max Queue Lengths:")
        for color, length in summary["max_queue_length_by_color"].items():
            print(f"  - {color.capitalize()}: {length}")
        print("--------------------------")

    def plot(self):
        """
        Generates and displays plots from the collected simulation metrics.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        print("\n--- Generating plots ---")
        fig, axs = plt.subplots(8, 1, figsize=(15, 30), sharex=True)
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

        # Plot 3: Gridlock Events
        ax3 = axs[2]
        if self.gridlock_events:
            times, is_stuck = zip(*self.gridlock_events)
            ax3.plot(
                times,
                [int(s) for s in is_stuck],
                label="Gridlock State",
                color="orange",
                drawstyle="steps-post",
            )
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["OK", "Gridlocked"])
            ax3.set_ylabel("Queue State")
            ax3.set_title("Interplay: Gridlock / Queue Starvation Events")
            ax3.legend()
            ax3.grid(True)

        # Plot 4: Age of Oldest Job in Queue
        ax4 = axs[3]
        if self.oldest_job_age:
            times, ages = zip(*self.oldest_job_age)
            ax4.plot(times, ages, label="Age of Oldest Job", color="brown")
            ax4.set_ylabel("Wait Time (steps)")
            ax4.set_title("Interplay: Age of Oldest Pending Job (Starvation)")
            ax4.legend()
            ax4.grid(True)

        # Plot 5: Job Flow Rates (Queue Stability)
        ax5 = axs[4]
        if self.job_submission_times and self.job_start_times:
            max_time = max(
                self.job_submission_times[-1],
                max(t for _, t in self.job_start_times),
            )
            bins = np.arange(0, max_time + 10, 10)
            sub_counts, _ = np.histogram(self.job_submission_times, bins=bins)
            start_times_only = [t for _, t in self.job_start_times]
            start_counts, _ = np.histogram(start_times_only, bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax5.plot(
                bin_centers,
                sub_counts,
                label="Submission Rate (jobs/10s)",
                color="blue",
            )
            ax5.plot(
                bin_centers, start_counts, label="Start Rate (jobs/10s)", color="green"
            )
            ax5.set_ylabel("Job Rate")
            ax5.set_title("Queue Stability: Job Submission vs. Start Rates")
            ax5.legend()
            ax5.grid(True)

        # Plot 6: Instance Demand vs. Provisioned Capacity (Control System Stability)
        ax6 = axs[5]
        if self.instance_demand_over_time and self.instance_counts:
            demand_times, demand_values = zip(*self.instance_demand_over_time)
            ax6.plot(
                demand_times, demand_values, label="Instance Demand", color="purple"
            )
            prov_times, prov_values, _, _ = zip(*self.instance_counts)
            ax6.plot(
                prov_times,
                prov_values,
                label="Provisioned Capacity",
                color="black",
                linestyle="--",
            )
            ax6.set_ylabel("Number of Instances")
            ax6.set_title("Control System Stability: Demand vs. Capacity")
            ax6.legend()
            ax6.grid(True)

        # Plot 7: Pool Utilization
        ax7 = axs[6]
        if self.pool_utilization:
            times, utils = zip(*self.pool_utilization)
            ax7.plot(times, [u * 100 for u in utils], label="Utilization %")
            ax7.axhline(y=85, color="r", linestyle="--", label="High Threshold (85%)")
            ax7.axhline(y=60, color="g", linestyle="--", label="Low Threshold (60%)")
            ax7.set_ylabel("Utilization (%)")
            ax7.set_title("Instance Pool Utilization")
            ax7.set_ylim(0, 110)
            ax7.legend()
            ax7.grid(True)

        # Plot 8: Job Wait Time Distribution
        ax8 = axs[7]
        if self.job_wait_times:
            ax8.hist(self.job_wait_times, bins=30, alpha=0.75)
            avg_wait = self.get_average_job_wait_time()
            ax8.axvline(
                avg_wait,
                color="r",
                linestyle="dashed",
                linewidth=2,
                label=f"Avg Wait: {avg_wait:.2f}",
            )
            ax8.set_xlabel("Time")
            ax8.set_ylabel("Number of Jobs")
            ax8.set_title("Job Wait Time Distribution")
            ax8.legend()
            ax8.grid(True)

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

            all_jobs_in_queue = self.job_queue.get_all_jobs()

            for job in all_jobs_in_queue:
                self.metrics.record_job_submission(job)

            instance_demand = sum(job.instances_required for job in all_jobs_in_queue)
            self.metrics.record_instance_demand(self.env.now, instance_demand)

            # --- New Interplay Metrics ---

            # 1. Record Gridlock State
            has_pending = bool(all_jobs_in_queue)
            schedulable_job_found = (
                self.job_queue.find_schedulable_job(self.instance_pool) is not None
            )
            is_stuck = has_pending and not schedulable_job_found
            self.metrics.record_gridlock_event(self.env.now, is_stuck)

            # 2. Record Age of Oldest Job
            if all_jobs_in_queue:
                oldest_job = min(all_jobs_in_queue, key=lambda j: j.submission_time)
                age = self.env.now - oldest_job.submission_time
                self.metrics.record_oldest_job_age(self.env.now, age)
            else:
                self.metrics.record_oldest_job_age(
                    self.env.now, 0
                )  # Record 0 if queue is empty

            # --- End New Interplay Metrics ---

            yield self.env.timeout(1)

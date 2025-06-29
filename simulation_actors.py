import simpy
import random


# Import components from the other file
from metrics import MetricsCollector
from simulation_components import (
    Job,
    JobQueue,
    InstancePool,
    EC2Service,
    Instance,
)


class Submitter:
    """
    Creates and submits jobs to the JobQueue at regular intervals, simulating user load.
    Corresponds to the "Submitters" actor in the specification.
    """

    def __init__(
        self,
        env: simpy.Environment,
        job_queue: JobQueue,
        color: str,
        max_jobs_per_tick: int = 2,
        max_instances_per_job: int = 10,
        tags: dict[str, str] | None = None,
    ):
        self.env = env
        self.job_queue = job_queue
        self.color = color
        self.max_jobs_per_tick = max_jobs_per_tick
        self.max_instances_per_job = max_instances_per_job
        self.tags = tags if tags is not None else {}
        self.action = env.process(self.run())

    def run(self):
        """The main simulation process for the Submitter."""
        while True:
            # At each time step, create a variable number of jobs
            num_jobs_to_create = random.randint(0, self.max_jobs_per_tick)

            for _ in range(num_jobs_to_create):
                instances_req = random.randint(1, self.max_instances_per_job)
                runtime = random.uniform(10, 50)

                job = Job(
                    color=self.color,
                    instances_required=instances_req,
                    runtime=runtime,
                    tags=self.tags,
                    submission_time=self.env.now,
                )
                self.job_queue.put(job)

            # Wait for the next time step
            yield self.env.timeout(1)


class JobScheduler:
    """
    Monitors the JobQueue and schedules jobs onto the InstancePool using backfilling.
    Corresponds to the "Job Scheduler" actor in the specification.
    """

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
        """
        The main scheduling loop, implementing backfilling.

        NOTE: This implementation uses a simple "EASY backfilling" approach, where
        it schedules the first available job that fits. While this keeps
        utilization high, it is not optimal for the goal of minimizing average
        wait time, as it can lead to starvation for large, high-priority jobs.
        A more advanced "Conservative Backfilling" strategy would provide
        reservations for high-priority jobs to prevent this.
        """
        while True:
            # Continuously try to schedule jobs as long as there are schedulable ones
            while True:
                # Find the highest-priority job that can run now (backfilling)
                schedulable_job = self.job_queue.find_schedulable_job(
                    self.instance_pool
                )

                if schedulable_job:
                    # Remove the job from the queue
                    job = self.job_queue.remove_job(schedulable_job.id)
                    if not job:
                        continue  # Job was removed elsewhere, try again

                    # Allocate instances
                    allocated_instances = self.instance_pool.get_matching_instances(job)

                    if allocated_instances:
                        self.metrics.record_job_start(job, self.env.now)
                        # Start a separate process to manage the job's lifecycle
                        self.env.process(self._run_job(job, allocated_instances))
                    else:
                        # This case is rare but could happen in a race condition.
                        # Re-queue the job and break to the next time step.
                        self.job_queue.put(job)
                        break
                else:
                    # No schedulable jobs found, break the inner loop
                    break

            # Wait for the next scheduling cycle
            yield self.env.timeout(1)

    def _run_job(self, job: Job, allocated_instances: list[Instance]):
        """A process that represents a job running on instances."""
        yield self.env.timeout(job.runtime)

        # Job finished, release instances and record metrics
        self.instance_pool.release_instances(allocated_instances)
        self.metrics.record_job_completion(job, self.env.now)


class PoolManager:
    """
    Manages the size of the InstancePool based on utilization thresholds and queue demand.
    Corresponds to the "Pool Manager" actor in the specification.

    Its scaling logic is designed to be robust, handling not just utilization but
    also scenarios like a "cold start" (scaling from zero instances) and
    "large job starvation" (ensuring the pool grows to accommodate large jobs).
    """

    def __init__(
        self,
        env: simpy.Environment,
        instance_pool: InstancePool,
        job_queue: JobQueue,
        ec2_service: EC2Service,
        high_threshold: float = 0.90,
        low_threshold: float = 0.70,
        scale_up_cooldown: int = 3,
        scale_down_cooldown: int = 5,
        scale_up_amount: int = 10,
        scale_down_amount: int = 5,
    ):
        self.env = env
        self.instance_pool = instance_pool
        self.job_queue = job_queue
        self.ec2_service = ec2_service
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.scale_up_amount = scale_up_amount
        self.scale_down_amount = scale_down_amount
        self.time_above_threshold = 0
        self.time_below_threshold = 0
        self.action = env.process(self.run())

    def run(self):
        """
        The main pool management loop.

        This method implements an advanced scaling logic that considers three
        primary signals to make scaling decisions, making it robust against
        common issues like cold starts and large-job starvation.
        """
        while True:
            utilization = self.instance_pool.get_utilization()
            total_instances = self.instance_pool.get_total_instances()

            # --- Advanced Scaling Logic ---
            # The decision to scale up is based on a combination of reactive and
            # proactive signals to ensure both efficiency and throughput.

            # Signal 1: Gridlock (Stuck Queue).
            # This occurs when jobs are waiting but none can be scheduled on the
            # currently free instances. It's a clear sign that more resources
            # are needed, regardless of current utilization. This is critical
            # for scaling up from zero instances (cold start).
            has_pending_jobs = bool(self.job_queue)
            is_schedulable_job_found = (
                self.job_queue.find_schedulable_job(self.instance_pool) is not None
            )
            is_stuck = has_pending_jobs and not is_schedulable_job_found

            # Signal 2: Insufficient Capacity (Proactive).
            # This checks if the pool is fundamentally too small to run the
            # largest waiting job. This prevents starvation where a large job
            # is perpetually blocked by a stream of smaller jobs that keep
            # utilization high but never allow the pool to grow enough.
            largest_job = self.job_queue.get_largest_pending_job()
            insufficient_capacity = (
                largest_job is not None
                and total_instances < largest_job.instances_required
            )

            # --- Define Scaling Conditions ---
            # Scale up if utilization is high (reactive), OR if the queue is
            # stuck, OR if the pool is too small for a waiting job (proactive).
            should_scale_up = (
                utilization > self.high_threshold or is_stuck or insufficient_capacity
            )

            # Scale down only if utilization is low AND we are not in a state
            # that requires scaling up. The `not should_scale_up` check is crucial
            # to prevent terminating instances when low utilization is actually
            # a symptom of insufficient capacity for a large waiting job.
            should_scale_down = (
                utilization < self.low_threshold
                and total_instances > 0
                and not should_scale_up
            )

            # --- Update Cooldown Timers based on conditions ---
            # Cooldown timers prevent the system from "flapping" (scaling up and
            # down too frequently) by requiring a condition to persist for a
            # few time steps before acting.
            if should_scale_up:
                self.time_above_threshold += 1
                self.time_below_threshold = 0
            elif should_scale_down:
                self.time_below_threshold += 1
                self.time_above_threshold = 0
            else:  # In the sweet spot or an ambiguous state, reset timers
                self.time_above_threshold = 0
                self.time_below_threshold = 0

            # --- Perform Scaling Actions ---
            if self.time_above_threshold >= self.scale_up_cooldown:
                new_instances: list[Instance] = yield self.env.process(
                    self.ec2_service.provision_instances(self.scale_up_amount, tags={})
                )
                self.instance_pool.add_instances(new_instances)
                self.time_above_threshold = 0  # Reset counter after scaling

            elif self.time_below_threshold >= self.scale_down_cooldown:
                idle_instances = self.instance_pool.free_instances
                num_to_terminate = min(len(idle_instances), self.scale_down_amount)
                if num_to_terminate > 0:
                    instances_to_terminate = idle_instances[:num_to_terminate]
                    yield self.env.process(
                        self.ec2_service.deprovision_instances(instances_to_terminate)
                    )
                    self.instance_pool.remove_instances(instances_to_terminate)
                self.time_below_threshold = 0  # Reset counter after scaling

            # Wait for the next time step
            yield self.env.timeout(1)

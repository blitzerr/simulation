from random import randint
from typing import Generator
import simpy
import collections
import heapq
import itertools  # For unique job IDs


class Job:
    """
    Represents a single job to be executed.
    """

    _id_counter = itertools.count()  # For unique job IDs

    def __init__(
        self,
        color: str,
        instances_required: int,
        runtime: float,
        tags: dict[str, str],
        submission_time: float,
    ):
        """Initializes a new Job object.

        Args:
            color: An identifier for the submitter of the job.
            instances_required: The number of instances required to run the job.
            runtime: The time it takes for the job to complete once started.
            tags: A dictionary of key/value pairs that must match instance tags.
            submission_time: The simulation time at which the job was submitted.
        """
        self.id = next(Job._id_counter)
        self.color = color
        self.instances_required = instances_required
        self.runtime = runtime
        self.tags = tags
        self.submission_time = submission_time
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.wait_time: float | None = None  # Calculated after job completes

    def __repr__(self):
        return (
            f"Job(id={self.id}, color='{self.color}', instances_required={self.instances_required}, "
            f"runtime={self.runtime:.2f}, tags={self.tags}, submitted={self.submission_time:.2f})"
        )


class Instance:
    """
    Represents a single compute instance.
    """

    _id_counter = itertools.count()  # For unique instance IDs

    def __init__(self, tags: dict[str, str]):
        self.id = next(Instance._id_counter)
        self.tags = tags
        self.is_busy = False  # True if currently running a job
        self.current_job_id: int | None = (
            None  # ID of the job currently running on this instance
        )

    def __repr__(self):
        return (
            f"Instance(id={self.id}, tags={self.tags}, busy={self.is_busy}, "
            f"job={self.current_job_id if self.is_busy else 'None'})"
        )


class InstancePool:
    """
    Manages a collection of Instance objects.
    Will provide methods for getting/releasing instances and checking utilization.
    """

    def __init__(self, env: simpy.Environment):
        self.env = env
        self.free_instances: list[Instance] = []
        self.busy_instances: list[Instance] = []
        self.all_instances: dict[int, Instance] = {}  # {instance_id: Instance_object}

    def can_satisfy_job(self, job: Job) -> bool:
        """
        Checks if there are enough free instances with matching tags to satisfy the job.
        """
        # print(f"InstancePool: Checking if can satisfy job {job.id} (needs {job.instance_required} instances, tags={job.tags})")

        # Count available instances that match the job's tags
        matching_free_instances_count = 0
        for instance in self.free_instances:
            # Check if all job tags match instance tags
            tags_match = True
            for job_tag_key, job_tag_value in job.tags.items():
                if instance.tags.get(job_tag_key) != job_tag_value:
                    tags_match = False
                    break
            if tags_match:
                matching_free_instances_count += 1

        return matching_free_instances_count >= job.instances_required

    def get_matching_instances(self, job: Job) -> list[Instance]:
        """
        Attempts to get the required number of instances for a job.
        Assumes can_satisfy_job has already returned True.
        """
        allocated_instances: list[Instance] = []

        # Iterate through free instances and allocate matching ones
        # We iterate over a copy to allow modification of self.free_instances
        for instance in list(self.free_instances):
            if len(allocated_instances) >= job.instances_required:
                break  # Got enough instances

            tags_match = True
            for job_tag_key, job_tag_value in job.tags.items():
                if instance.tags.get(job_tag_key) != job_tag_value:
                    tags_match = False
                    break

            if tags_match:
                allocated_instances.append(instance)
                # Mark as busy and assign job_id
                instance.is_busy = True
                instance.current_job_id = job.id
                # Remove from free_instances (will do this outside loop)

        if len(allocated_instances) == job.instances_required:
            # Update the actual lists: move allocated instances from free to busy
            for instance in allocated_instances:
                self.free_instances.remove(instance)
                self.busy_instances.append(instance)
            # print(f"[{self.env.now:.2f}] InstancePool: Allocated {len(allocated_instances)} instances for job {job.id}.")
            return allocated_instances
        else:
            # This should ideally not happen if can_satisfy_job was called first
            # print(f"[{self.env.now:.2f}] InstancePool: Failed to allocate instances for job {job.id} despite check.")
            # Reset busy status if partial allocation happened
            for instance in allocated_instances:
                instance.is_busy = False
                instance.current_job_id = None
            return []

    def release_instances(self, instances: list[Instance]):
        """
        Releases instances back to the free pool after a job completes.
        """
        for instance in instances:
            if instance in self.busy_instances:
                self.busy_instances.remove(instance)
                instance.is_busy = False
                instance.current_job_id = None
                self.free_instances.append(instance)
            # else:
            # print(f"InstancePool: Warning: Attempted to release instance {instance.id} not in busy pool.")
        # print(f"[{self.env.now:.2f}] InstancePool: Released {len(instances)} instances.")

    def add_instances(self, new_instances: list[Instance]):
        """Adds newly provisioned instances to the pool."""
        for instance in new_instances:
            self.free_instances.append(instance)
            self.all_instances[instance.id] = instance
        # print(f"[{self.env.now:.2f}] InstancePool: Added {len(new_instances)} new instances. Total: {len(self.all_instances)}")

    def remove_instances(self, instances_to_remove: list[Instance]):
        """Removes instances from the pool (e.g., for deprovisioning)."""
        for instance in instances_to_remove:
            if instance in self.free_instances:
                self.free_instances.remove(instance)
                self.all_instances.pop(instance.id, None)
            # else:
            # print(f"InstancePool: Warning: Attempted to remove instance {instance.id} not in free pool.")
        # print(f"[{self.env.now:.2f}] InstancePool: Removed {len(instances_to_remove)} instances. Total: {len(self.all_instances)}")

    def get_utilization(self) -> float:
        """Calculates current instance pool utilization."""
        total_instances = len(self.all_instances)
        if total_instances == 0:
            return 0.0
        return len(self.busy_instances) / total_instances

    def get_total_instances(self) -> int:
        return len(self.all_instances)

    def get_free_instances_count(self) -> int:
        return len(self.free_instances)

    def get_busy_instances_count(self) -> int:
        return len(self.busy_instances)


class JobQueue:
    """
    A priority queue for jobs, where priority is submission time (FIFO).
    Supports finding schedulable jobs and removing specific jobs.
    """

    def __init__(self):
        # Min-heap: (submission_time, job_id, job_object)
        # job_id is included to break ties and ensure stable sorting for heapq
        self._queue: list[tuple[float, int, Job]] = []
        # Dictionary for quick lookup and removal by job ID
        self._job_map: dict[int, Job] = {}  # {job_id: job_object}

    def put(self, job: Job):
        """Adds a job to the queue."""
        if job.id in self._job_map:
            raise ValueError(f"Job with ID {job.id} already exists in the queue.")

        heapq.heappush(self._queue, (job.submission_time, job.id, job))
        self._job_map[job.id] = job
        # print(f"[{job.submission_time:.2f}] JobQueue: Added {job.id} (color={job.color}) to queue. Queue size: {len(self._queue)}")

    def find_schedulable_job(self, instance_pool: InstancePool) -> Job | None:
        """
        Iterates through jobs in the queue (priority order) and returns the first
        job that can be scheduled given the current instance pool state.
        Does NOT remove the job from the queue.

        Args:
            instance_pool: An instance of InstancePool, used to check job schedulability.

        Returns:
            The Job object if a schedulable job is found, otherwise None.
        """
        # Create a temporary list to iterate through, to ensure priority order for checking
        # and to handle potential removal of jobs that are no longer valid (e.g., replaced)
        # Sorting the heap is O(N log N), which might be a bottleneck for very large queues.
        # An alternative for "not strict priority" could be to iterate _job_map.values()
        # and check, which is O(N) but doesn't guarantee checking higher priority first.
        # For now, we prioritize checking higher priority jobs first.
        temp_queue_items = sorted(self._queue)

        for _, _job_id, job in temp_queue_items:
            # Check if the job is still valid (not removed/replaced)
            if job.id not in self._job_map:
                continue  # This job was removed/replaced while iterating

            # Ask the instance pool if it can satisfy this job
            # instance_pool.can_satisfy_job needs to be implemented in InstancePool
            if instance_pool.can_satisfy_job(job):
                # print(f"[{instance_pool.env.now:.2f}] JobQueue: Found schedulable job {job.id} (color={job.color})")
                return job
        # print(f"[{instance_pool.env.now:.2f}] JobQueue: No schedulable job found.")
        return None

    def remove_job(self, job_id: int) -> Job | None:
        """Removes a job from the queue by its ID."""
        if job_id not in self._job_map:
            # print(f"JobQueue: Warning: Attempted to remove non-existent job ID {job_id}.")
            return

        removed = self._job_map.pop(job_id)
        # Note: This does not remove the job from the underlying heapq directly.
        # The find_schedulable_job method handles this by checking _job_map.
        # If the heap grows very large with 'stale' entries, a periodic rebuild
        # of the heap might be necessary (e.g., if len(_queue) > 2 * len(_job_map)).
        # print(f"JobQueue: Removed job {job_id}.")
        return removed

    def replace_job(self, old_job_id: int, new_job: Job):
        """
        Replaces an old job with a new one. The new job takes the priority of the old job.
        If the old job does not exist, it simply adds the new job.
        """
        old = self.remove_job(old_job_id)
        if old is not None:
            new_job.submission_time = old.submission_time
        self.put(new_job)
        # print(f"JobQueue: Replaced job {old_job_id} with new job {new_job.id}.")

    def get_queue_length_by_color(self) -> dict[str, int]:
        """Returns the current queue length broken down by job color."""
        lengths: dict[str, int] = collections.defaultdict(int)
        for job in self._job_map.values():  # Iterate through valid jobs
            lengths[job.color] += 1
        return dict(lengths)

    def get_all_jobs(self) -> list[Job]:
        """Returns a list of all jobs currently in the queue."""
        return list(self._job_map.values())

    def get_largest_pending_job(self) -> Job | None:
        """
        Finds the job in the queue that requires the most instances.
        
        This is a key method for the PoolManager's proactive scaling logic,
        allowing it to ensure the pool is large enough for even the biggest
        waiting jobs.
        
        Returns:
            The Job object with the highest instance requirement, or None if the queue is empty.
        """
        # This is an O(N) operation where N is the number of jobs in the queue.
        # For a very large queue, this could be optimized if needed.
        if not self._job_map:
            return None

        return max(self._job_map.values(), key=lambda job: job.instances_required)

    def __len__(self):
        return len(self._job_map)

    def __bool__(self):
        return len(self._job_map) > 0


class EC2Service:
    """
    Simulates the EC2 service for provisioning and de-provisioning instances.
    """

    def __init__(
        self,
        env: simpy.Environment,
        provisioning_delay: float = 5.0,
        deprovisioning_delay: float = 2.0,
    ):
        self.env = env
        self.provisioning_delay = provisioning_delay
        self.deprovisioning_delay = deprovisioning_delay
        self._id_counter = itertools.count()  # For unique EC2 request IDs

    def provision_instances(
        self, num_asked: int, tags: dict[str, str] | None = None
    ) -> Generator[simpy.Timeout, None, list[Instance]]:
        """
        Simulates provisioning instances. Returns a process that yields the new instances.
        """
        _request_id: int = next(self._id_counter)
        # print(f"[{self.env.now:.2f}] EC2Service: Provisioning request {request_id} for {num_asked} instances (tags={tags}).")
        yield self.env.timeout(self.provisioning_delay)

        # EC2 can provide between 0 and asked number of instances.
        # For simplicity, let's assume it always provides the asked number for now.
        # This can be made more complex later (e.g., random failures, capacity limits).

        # a random number between 0 and num_asked
        actual_provisioned = randint(0, num_asked)  # Simulate

        new_instances = [
            Instance(tags if tags is not None else {})
            for _ in range(actual_provisioned)
        ]
        # print(f"[{self.env.now:.2f}] EC2Service: Provisioning request {request_id} completed. Provided {len(new_instances)} instances.")
        return new_instances

    def deprovision_instances(
        self, instances_to_return: list[Instance]
    ) -> Generator[simpy.Timeout, None, bool]:
        """
        Simulates de-provisioning instances. Returns a process that yields when complete.
        """
        _request_id = next(self._id_counter)
        # print(f"[{self.env.now:.2f}] EC2Service: Deprovisioning request {request_id} for {len(instances_to_return)} instances.")
        yield self.env.timeout(self.deprovisioning_delay)
        # In a real system, these instances would be truly gone. Here, we just simulate the time.
        # print(f"[{self.env.now:.2f}] EC2Service: Deprovisioning request {request_id} completed.")
        return True  # Indicate completion

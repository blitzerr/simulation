# Job Execution System Simulation: Specification

## System Overview

This document describes a simulation of a cloud-based job execution system. The system is designed to process jobs submitted by various users, manage a pool of compute instances, and schedule jobs onto those instances based on their requirements. The primary goal is to model the interplay between job submission rates, scheduling logic, and resource pool management.

---

## 1. Core Components (Actors)

*   **Submitters**:
    *   Entities that create and submit `Jobs` to the system.
    *   Each `Submitter` is identified by a unique `color` (e.g., red, blue), which is imparted to the jobs they create.
    *   At each discrete time step, a `Submitter` creates a variable number of jobs, `N`, determined
        by a function known only to that submitter. Each job requires some instance count between 1
        and K.

*   **Job Scheduler**:
    *   The central logic unit whose primary goal is to **minimize the average wait time** for all jobs.
    *   It pulls `Jobs` from the `Queue` and attempts to match them with available `Instances` from the `Instance Pool`.
    *   A job is scheduled only when all its requirements are met.

*   **Pool Manager**:
    *   Responsible for maintaining the health and utilization of the `Instance Pool`. Its primary goal is to keep pool utilization within a target range (e.g., 85-95%).
    *   It monitors pool utilization and scales the number of `Instances` up or down to meet its target.
    *   It interacts with `EC2` to provision or terminate instances.

*   **EC2 (Instance Provider)**:
    *   An external service that provides on-demand compute `Instances`.
    *   When the `Pool Manager` requests `N` instances, `EC2` may return anywhere between 0 and `N` instances.

---

## 2. System Objects

*   **Job**: A unit of work with the following properties:
    *   `color`: An identifier inherited from the `Submitter`.
    *   `instance_required`: The exact number of `Instances` required to run the job.
    *   `runtime`: The duration (in time steps) the `Instances` will be occupied by the job.
    *   `tags`: A set of key-value pairs. A job can only run on `Instances` that have a matching set of tags.

*   **Queue**:
    *   A priority queue that holds submitted `Jobs` awaiting execution.
    *   The default priority is the job's submission time, creating a FIFO (First-In, First-Out) behavior.
    *   Priority can be explicitly set to a time in the past, allowing a new job to effectively "jump the queue."

*   **Instance**:
    *   A virtual server that executes `Jobs`.
    *   Each `Instance` has a set of immutable key-value `tags` assigned at creation.

*   **Instance Pool**:
    *   A collection of all active `Instances` available for scheduling.
    *   This pool is managed exclusively by the `Pool Manager`.

---

## 3. System Dynamics & Scheduling Logic

1.  **Job Submission**: `Submitters` push new `Jobs` into the `Queue` at each time step.
2.  **Scheduling Cycle**: The `Job Scheduler` activates at each time step to perform a scheduling pass.
3.  **Scheduling for Minimal Wait Time (Backfilling)**: The scheduler's primary goal is to minimize the average job wait time. To achieve this, it employs a **backfilling** strategy. While jobs are sorted by priority, the scheduler will not let resources sit idle if the highest-priority job is too large to run. Instead, it will scan for lower-priority jobs that can fit into the available resources.
    *   **Example**: Job A (priority 1) requires 10 instances. Job B (priority 2) requires 1 instance. If the pool has only 5 free instances, the scheduler will skip Job A and schedule Job B. This ensures system progress and reduces the overall average wait time, even though Job A's individual wait time is extended.
4.  **Unscheduled Jobs**: If a job cannot be scheduled during a pass, it remains in the `Queue` with its original priority, waiting for the next scheduling cycle.

---

## 4. Analysis and Specification Refinements

To ensure the simulation is well-defined, the following logical points must be clarified.

*   **Scheduler-Queue Interaction**:
    The scheduler iterates through jobs in the queue in **strict priority order**. If the highest-priority job cannot be scheduled, it attempts to schedule the next-highest-priority job, and so on. This allows lower-priority jobs to run if resources for higher-priority jobs are not yet available.

*   **Pool Manager's Algorithm**:
    The `Pool Manager` uses a sophisticated, multi-faceted algorithm to prevent common failure modes like cold starts and large-job starvation. It scales up if **any** of the following conditions are met, subject to a cooldown period (`X` time steps):
        *   **High Utilization (Reactive)**: Pool-wide utilization exceeds a high threshold (e.g., 90%). This is the standard reactive scaling trigger.
        *   **Gridlock (Stuck Queue)**: There are jobs waiting in the queue, but none of them can be scheduled on the currently available free instances. This handles cases where the pool is too fragmented or all instances are busy, and it solves the "cold start" problem (scaling from 0 instances).
        *   **Insufficient Capacity (Proactive)**: The total number of instances in the pool is less than the number required by the largest job in the queue. This proactively scales the pool to ensure even very large jobs can eventually run, preventing starvation.

    It scales down only if utilization is low (e.g., below 70%) for a sustained period (`Y` time steps) **and** none of the scale-up conditions are met. This prevents the manager from terminating instances when low utilization is caused by a large job waiting for more capacity.

*   **Job Tag Matching Logic**:
    A job's tag requirements are met by a set of instances if, for every instance in that set, the instance's tags are a **superset** of the job's tags. This means an instance can have extra tags that the job does not require.

---

## 5. Initial Conditions

To ensure reproducibility, the simulation must start from a defined state:

*   **Time**: The simulation starts at `t = 0`.
*   **Queue**: The `Queue` is initially empty.
*   **Instance Pool**: The `Instance Pool` is initially empty.
*   **Submitters**: `Submitters` begin their job creation process at `t = 0`.

---

## 6. Key Metrics & Parameters

*   **Monitoring Metrics (Outputs)**:
    *   Queue length, categorized by `color`.
    *   `Instance Pool` utilization percentage.
    *   Average job wait time.

*   **Uncontrolled Parameters (Inputs)**:
    *   The rate and volume of jobs submitted by each `Submitter`.
    *   The `instance_required`, `runtime`, and `tags` for each job.
    *   If the instance provider will be able to service the request from the pool manager.

*   **Control Signal (System Lever)**:
    *   The number of new instances the `Pool Manager` requests from `EC2`.
    *   The scaling thresholds and parameters used by the `Pool Manager`.

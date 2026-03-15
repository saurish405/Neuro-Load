import simpy
import random

class Server:
    def __init__(self, env, id, speed):
        self.env = env
        self.id = id
        self.speed = speed  # Lower number = Faster processing
        self.resource = simpy.Resource(env, capacity=10) # Can handle 10 tasks at once
        self.tasks_processed = 0

    def handle_task(self, task_name):
        with self.resource.request() as req:
            yield req
            # Simulating processing time
            duration = random.expovariate(1.0 / self.speed)
            yield self.env.timeout(duration)
            self.tasks_processed += 1
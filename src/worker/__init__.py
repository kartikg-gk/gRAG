"""The compile path.

Separate from the serving path, including its configuration: the two run as
different processes and a deployment can have one without the other.

One piece so far — deciding when an organisation is due to be rebuilt.
Nothing compiles, enqueues, or reads a database yet.
"""

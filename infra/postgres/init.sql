-- Run by the Postgres image once, when its data volume is first created.
--
-- The platform keeps two databases on one server: the control plane
-- (organisations, keys, pods, jobs, artifacts, chat history) and the graph
-- store (every tenant's accumulated entities and relationships, which each
-- build reads in bulk). The application creates the tables; this only makes
-- the databases.
--
-- CREATE DATABASE cannot run inside a function or take IF NOT EXISTS, so each
-- statement is generated only when the database is missing and run with \gexec.

SELECT 'CREATE DATABASE control_plane'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'control_plane')\gexec

SELECT 'CREATE DATABASE graph_store'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'graph_store')\gexec

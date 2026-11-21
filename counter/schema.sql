-- The counters' table. Run once in the D1 database's console (see README.md).
CREATE TABLE IF NOT EXISTS plays (
  game TEXT PRIMARY KEY,
  n    INTEGER NOT NULL DEFAULT 0
);

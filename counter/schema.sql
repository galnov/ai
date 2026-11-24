-- The counters' table. Run once in the D1 database's console (see README.md).
CREATE TABLE IF NOT EXISTS plays (
  game TEXT PRIMARY KEY,
  n    INTEGER NOT NULL DEFAULT 0
);

-- the leaderboards
-- a game that was started and not submitted yet (one row per game; deleted when it is submitted)
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, game TEXT NOT NULL, board TEXT NOT NULL,
  seed INTEGER NOT NULL, opts TEXT NOT NULL DEFAULT '{}', created INTEGER NOT NULL);
-- every checked score
CREATE TABLE IF NOT EXISTS scores (id INTEGER PRIMARY KEY AUTOINCREMENT, game TEXT NOT NULL, board TEXT NOT NULL,
  name TEXT NOT NULL, score REAL NOT NULL, created INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS scores_board ON scores (game, board, score);

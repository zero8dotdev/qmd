import { afterEach, beforeEach, expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { addMessage, initializeMemoryTables } from "./memory";

let db: Database;

beforeEach(() => {
  db = new Database(":memory:");
  db.exec("PRAGMA foreign_keys = ON");
  initializeMemoryTables(db);
});

afterEach(() => {
  db.close();
});

test("addMessage creates session once and does not overwrite existing title", async () => {
  await addMessage(db, "s1", "user", "first", { title: "Initial Title" });
  await addMessage(db, "s1", "assistant", "second", { title: "New Title" });

  const session = db
    .prepare(`SELECT id, title FROM memory_sessions WHERE id = ?`)
    .get("s1") as { id: string; title: string } | null;
  const sessionCount = (
    db.prepare(`SELECT COUNT(*) AS count FROM memory_sessions WHERE id = ?`).get("s1") as {
      count: number;
    }
  ).count;
  const messageCount = (
    db.prepare(`SELECT COUNT(*) AS count FROM memory_messages WHERE session_id = ?`).get("s1") as {
      count: number;
    }
  ).count;

  expect(session).not.toBeNull();
  expect(session?.title).toBe("Initial Title");
  expect(sessionCount).toBe(1);
  expect(messageCount).toBe(2);
});

test("addMessage backfills empty title when provided later", async () => {
  const now = new Date().toISOString();
  db.prepare(
    `INSERT INTO memory_sessions (id, title, created_at, updated_at, active) VALUES (?, ?, ?, ?, 1)`
  ).run("s-empty", "", now, now);

  await addMessage(db, "s-empty", "user", "hello", { title: "Backfilled Title" });

  const session = db
    .prepare(`SELECT title FROM memory_sessions WHERE id = ?`)
    .get("s-empty") as { title: string } | null;
  expect(session?.title).toBe("Backfilled Title");
});

test("addMessage with sessionId 'new' generates a concrete session id", async () => {
  const message = await addMessage(db, "new", "user", "hello");
  expect(message.session_id).not.toBe("new");
  expect(message.session_id.length).toBe(8);

  const session = db
    .prepare(`SELECT id FROM memory_sessions WHERE id = ?`)
    .get(message.session_id) as { id: string } | null;
  expect(session?.id).toBe(message.session_id);
});

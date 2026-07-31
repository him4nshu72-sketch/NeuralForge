import {
  integer,
  jsonb,
  pgTable,
  real,
  serial,
  text,
  timestamp,
} from "drizzle-orm/pg-core";

/**
 * Every training run a user saves from the lab becomes an "experiment".
 * The full hyper-parameter configuration is stored so a run can be
 * reproduced exactly, plus a tiny decision-boundary thumbnail so the
 * leaderboard can render what the model actually learned.
 */
export const experiments = pgTable("experiments", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  author: text("author").notNull().default("anonymous"),

  // dataset
  dataset: text("dataset").notNull(),
  noise: real("noise").notNull().default(0),
  samples: integer("samples").notNull().default(400),

  // architecture / hyper-parameters
  hidden: jsonb("hidden").$type<number[]>().notNull().default([]),
  features: jsonb("features").$type<string[]>().notNull().default([]),
  activation: text("activation").notNull(),
  optimizer: text("optimizer").notNull(),
  learningRate: real("learning_rate").notNull(),
  batchSize: integer("batch_size").notNull(),
  l2: real("l2").notNull().default(0),

  // results
  epochs: integer("epochs").notNull(),
  paramCount: integer("param_count").notNull(),
  trainLoss: real("train_loss").notNull(),
  testLoss: real("test_loss").notNull(),
  trainAcc: real("train_acc").notNull(),
  testAcc: real("test_acc").notNull(),

  // visualisations
  history: jsonb("history")
    .$type<{ epoch: number; trainLoss: number; testLoss: number; testAcc: number }[]>()
    .notNull()
    .default([]),
  thumb: jsonb("thumb")
    .$type<{ size: number; classes: number[]; conf: number[] }>()
    .notNull(),

  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export type Experiment = typeof experiments.$inferSelect;
export type NewExperiment = typeof experiments.$inferInsert;

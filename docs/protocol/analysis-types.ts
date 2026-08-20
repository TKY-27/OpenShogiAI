export const ANALYSIS_SCHEMA = "open_shogi_analysis/v1" as const;
export const TIME_CONTROL_SCHEMA = "open_shogi_time_control/v1" as const;

export interface AnalysisStart {
  schema: typeof ANALYSIS_SCHEMA;
  positionSfen: string;
  modelHash: string;
  evaluatorConfigHash: string;
  featureSchemaHash: string;
  evaluationSemanticsHash: string;
  searchOptionsHash: string;
  openingProfileHash: string;
  multiPv: number;
}

export interface AnalysisStep {
  schema: typeof ANALYSIS_SCHEMA;
  nodes: number;
  maxDepth: number;
  timestampMs: number;
}

export interface AnalysisLine {
  rank: number;
  score: number;
  mateScore: number | null;
  depth: number;
  nodes: number;
  pv: string[];
}

export interface AnalysisUpdate {
  source: "cache" | "search";
  canonicalPosition: string;
  positionHash: string;
  modelHash: string;
  evaluatorConfigHash: string;
  featureSchemaHash: string;
  evaluationSemanticsHash: string;
  searchOptionsHash: string;
  openingProfileHash: string;
  multiPv: number;
  depth: number;
  nodes: number;
  nps: number;
  score: number;
  mateScore: number | null;
  lines: AnalysisLine[];
  timestampMs: number;
  engineVersion: string;
}

export interface AnalysisResponse {
  schema: typeof ANALYSIS_SCHEMA;
  event: string;
  updates: AnalysisUpdate[];
}

export interface AnalysisTransport {
  start(profile: string, evaluator: string, request: AnalysisStart): Promise<AnalysisResponse>;
  step(request: AnalysisStep): Promise<AnalysisResponse>;
  stop(): Promise<AnalysisResponse>;
}

/** Minimal cooperative client: cached display first, then bounded renewed search slices. */
export class AnalysisClient {
  private running = false;

  constructor(
    private readonly transport: AnalysisTransport,
    private readonly publish: (update: AnalysisUpdate) => void,
  ) {}

  async start(profile: string, evaluator: string, request: AnalysisStart): Promise<void> {
    this.running = true;
    const started = await this.transport.start(profile, evaluator, request);
    started.updates.forEach(this.publish);
  }

  async runSlice(nodes: number, maxDepth: number): Promise<void> {
    if (!this.running) return;
    const response = await this.transport.step({
      schema: ANALYSIS_SCHEMA,
      nodes,
      maxDepth,
      timestampMs: Date.now(),
    });
    response.updates.forEach(this.publish);
  }

  async stop(): Promise<void> {
    this.running = false;
    await this.transport.stop();
  }
}

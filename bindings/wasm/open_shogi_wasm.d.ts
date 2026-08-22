/* tslint:disable */
/* eslint-disable */

export class WasmBrowserEngine {
    free(): void;
    [Symbol.dispose](): void;
    analysisRestart(): string;
    analysisStart(profile: string, evaluator: string, request_json: string): string;
    analysisStep(request_json: string): string;
    analysisStop(): string;
    analysisWorkerFailed(): string;
    configureOpening(profile: string, max_plies: number, minimum_sample_count: bigint, maximum_teacher_loss_cp: number): string;
    loadModel(bytes: Uint8Array, expected_artifact_sha256?: string | null): string;
    loadOpeningBook(bytes: Uint8Array, expected_artifact_sha256?: string | null): string;
    constructor();
    playMove(movement: string): string;
    reset(sfen?: string | null): string;
    restore(initial_sfen: string, moves_json: string): string;
    search(profile: string, evaluator: string, multi_pv: number): string;
    searchWithTimeControl(profile: string, evaluator: string, multi_pv: number, time_control_json: string): string;
    snapshot(): string;
    unloadModel(): string;
    unloadOpeningBook(): string;
}

export class WasmOsaval02Model {
    free(): void;
    [Symbol.dispose](): void;
    deterministicTest(sfen: string, history_json?: string | null): string;
    identity(): string;
    infer(sfen: string, history_json?: string | null): string;
    constructor(bytes: Uint8Array);
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmbrowserengine_free: (a: number, b: number) => void;
    readonly __wbg_wasmosaval02model_free: (a: number, b: number) => void;
    readonly wasmbrowserengine_analysisRestart: (a: number) => [number, number, number, number];
    readonly wasmbrowserengine_analysisStart: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly wasmbrowserengine_analysisStep: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmbrowserengine_analysisStop: (a: number) => [number, number, number, number];
    readonly wasmbrowserengine_analysisWorkerFailed: (a: number) => [number, number, number, number];
    readonly wasmbrowserengine_configureOpening: (a: number, b: number, c: number, d: number, e: bigint, f: number) => [number, number, number, number];
    readonly wasmbrowserengine_loadModel: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmbrowserengine_loadOpeningBook: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmbrowserengine_new: () => number;
    readonly wasmbrowserengine_playMove: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmbrowserengine_reset: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmbrowserengine_restore: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmbrowserengine_search: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly wasmbrowserengine_searchWithTimeControl: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly wasmbrowserengine_snapshot: (a: number) => [number, number, number, number];
    readonly wasmbrowserengine_unloadModel: (a: number) => [number, number, number, number];
    readonly wasmbrowserengine_unloadOpeningBook: (a: number) => [number, number, number, number];
    readonly wasmosaval02model_deterministicTest: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmosaval02model_identity: (a: number) => [number, number, number, number];
    readonly wasmosaval02model_infer: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmosaval02model_new: (a: number, b: number) => [number, number, number];
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

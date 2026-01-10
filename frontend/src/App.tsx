import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, getJSON, postForm, postFormBlob } from "./api/client";
import type { MethodsResp, Report, AttributionStage } from "./types/report";

import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { HelpCircle } from "lucide-react";
import { Tooltip } from "@/components/ui/tooltip";

import { zipSync, strToU8, unzipSync, strFromU8 } from "fflate";

const META_KEY = "combined_methods";
const META_FAST_KEY = "combined_methods_fast";
const META_ONLY_KEY = "meta_model_only";
const META_LEGACY_KEY = "all_methods"; // backward compatibility
const MAX_DISPLAY_WIDTH = 512;


const HOW_META_ITEMS: Array<{ title: string; text: string }> = [
  {
    title: "Meta-model",
    text: `
    The meta-model is a trained classifier that combines the outputs of multiple
    detection methods into a single final decision.

    Each detector contributes its AI likelihood, and the meta-model learns how to
    weight and interpret them together, including handling missing or unreliable
    signals (for example, face-only models when no face is present).
    `.trim(),
  },
  {
    title: "Combined Methods (Gated Meta Pipeline)",
    text: `
    Combined methods use a staged pipeline around the meta-model.

    First, metadata and camera-origin checks are applied when available
    (such as C2PA or PRNU) to catch strong signals early.

    If no definitive gate is triggered, the system runs a set of base detectors
    and passes their outputs to the meta-model, which produces the final AI
    or REAL decision.
    `.trim(),
  },
];

function b64pngToDataUrl(b64: string) {
  return `data:image/png;base64,${b64}`;
}

function clamp01(x: number) {
  return Math.min(1, Math.max(0, x));
}

function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

function isNullish(x: unknown): x is null | undefined {
  return x === null || x === undefined;
}

function hasOwn(obj: any, key: string) {
  return obj && Object.prototype.hasOwnProperty.call(obj, key);
}

function fmtConfidence(x: number | null | undefined) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  return x.toFixed(2);
}

function isZipFile(f: File | null) {
  if (!f) return false;
  const name = (f.name || "").toLowerCase();
  return name.endsWith(".zip");
}

function u8ToBlobPart(u8: Uint8Array): ArrayBuffer {
  return u8.slice().buffer;
}


const INVALID_CHARS = /[^a-zA-Z0-9._\-]+/g;

function safeName(s: string) {
  let out = (s ?? "").trim().replace(/\\/g, "_").replace(/\//g, "_");
  out = out.replace(INVALID_CHARS, "_").replace(/^_+|_+$/g, "");
  return out || "unnamed";
}

function humanizeErrorMessage(e: any): string {
  if (e && typeof e === "object" && typeof e.detail === "string") {
    return e.detail;
  }

  const msg = e?.message ?? String(e ?? "");

  try {
    const obj = JSON.parse(msg);
    if (obj && typeof obj.detail === "string") return obj.detail;
  } catch {
    // ignore
  }

  return msg;
}

function makeLightReportFrontend(report: Report) {
  const out: any = { ...(report as any) };

  delete out.visuals;

  const attr = out.attribution;
  if (attr && typeof attr === "object") {
    const attr2: Record<string, any> = {};
    for (const [stageName, stage] of Object.entries(attr)) {
      if (stage && typeof stage === "object") {
        const stageCopy: any = { ...(stage as any) };
        delete stageCopy.visuals;
        attr2[stageName] = stageCopy;
      } else {
        attr2[stageName] = stage;
      }
    }
    out.attribution = attr2;
  }

  return out;
}

function b64ToBytes(b64: string): Uint8Array {
  const cleaned = b64.startsWith("data:") ? (b64.split(",")[1] ?? "") : b64;

  const bin = atob(cleaned);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function buildVisualsZipFrontend(report: Report): Blob {
  const files: Record<string, Uint8Array> = {};
  let count = 0;

  const visuals: any = (report as any)?.visuals;
  if (visuals && typeof visuals === "object") {
    for (const [k, b64] of Object.entries(visuals)) {
      if (typeof b64 === "string" && b64) {
        files[`visuals/${safeName(k)}.png`] = b64ToBytes(b64);
        count++;
      }
    }
  }

  const attr: any = (report as any)?.attribution;
  if (attr && typeof attr === "object") {
    for (const [stageName, stage] of Object.entries(attr)) {
      if (!stage || typeof stage !== "object") continue;

      const stageVisuals = (stage as any).visuals;
      if (!stageVisuals || typeof stageVisuals !== "object") continue;

      const stageFolder = safeName(String(stageName));
      for (const [k, b64] of Object.entries(stageVisuals)) {
        if (typeof b64 === "string" && b64) {
          files[`attribution/${stageFolder}/${safeName(k)}.png`] = b64ToBytes(
            b64
          );
          count++;
        }
      }
    }
  }

  if (count === 0) {
    files["README.txt"] = strToU8("No visualizations were present in the report.\n");
  }

  const zipped = zipSync(files, { level: 6 });
  const ab = zipped.slice().buffer;
  return new Blob([ab], { type: "application/zip" });
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}


type BatchCounts = { real: number; ai: number; unknown: number };

function blobToU8(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(new Uint8Array(r.result as ArrayBuffer));
    r.onerror = () => reject(r.error);
    r.readAsArrayBuffer(blob);
  });
}

function pickZipEntry(files: Record<string, Uint8Array>, name: string) {
  return files[name] ?? null;
}


function AttributionBlock({
  title,
  stage,
}: {
  title: string;
  stage: AttributionStage | undefined;
}) {
  if (!stage) return null;

  const { label, confidence, metrics, visuals } = stage;

  const visualsEntries = Object.entries(visuals || {}).sort(([a], [b]) =>
    a.localeCompare(b)
  );

  return (
    <div className="rounded-md border bg-white p-4 space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <div className="text-sm font-semibold">{title}</div>
          <div className="text-sm">
            <span className="font-medium">Label:</span>{" "}
            <span className="font-semibold">{label ?? "—"}</span>
          </div>
        </div>

        <div className="text-sm text-muted-foreground">
          confidence={fmtConfidence(confidence)}
        </div>
      </div>

      {/* Optional metrics */}
      {metrics && Object.keys(metrics).length > 0 && (
        <details className="rounded-md border bg-white p-3">
          <summary className="cursor-pointer text-sm font-medium">
            Attribution metrics
          </summary>
          <pre className="mt-3 max-h-[240px] overflow-auto rounded-md border bg-[#f7f9fb] p-3 text-xs">
            {JSON.stringify(metrics, null, 2)}
          </pre>
        </details>
      )}

      {/* Optional visuals */}
      {visualsEntries.length > 0 && (
        <div className="space-y-3">
          <div className="text-sm font-medium">Attribution visualizations</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {visualsEntries.map(([k, b64]) => (
              <div key={k} className="space-y-2">
                <div className="text-xs font-semibold">{k}</div>
                <img
                  src={b64pngToDataUrl(b64 as any)}
                  alt={k}
                  className="w-full h-auto"
                  style={{ maxWidth: MAX_DISPLAY_WIDTH }}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [methods, setMethods] = useState<MethodsResp["methods"]>([]);

  const [selected, setSelected] = useState<string[]>([META_KEY]);
  const [threshold, setThreshold] = useState<number>(0.5);

  const [file, setFile] = useState<File | null>(null);
  const [report, setReport] = useState<Report | null>(null);

  const [batchZip, setBatchZip] = useState<Blob | null>(null);

  const [batchResultsCsv, setBatchResultsCsv] = useState<Blob | null>(null);
  const [batchSummaryJson, setBatchSummaryJson] = useState<Blob | null>(null);
  const [batchPiePng, setBatchPiePng] = useState<Blob | null>(null);

  const [batchCounts, setBatchCounts] = useState<BatchCounts | null>(null);
  const [batchPieUrl, setBatchPieUrl] = useState<string | null>(null);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [methodsError, setMethodsError] = useState<string | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file || isZipFile(file)) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    getJSON<MethodsResp>("/methods")
      .then((d) => {
        setMethods(d.methods);

        setSelected((prev) => {
          const s = new Set(prev);
          if (s.has(META_LEGACY_KEY)) {
            s.delete(META_LEGACY_KEY);
            s.add(META_KEY);
          }
          return Array.from(s);
        });
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const howItems = useMemo(() => {
    const base = (methods as any[])
      .filter((m) => typeof m?.how_text === "string" && m.how_text.trim())
      .map((m) => ({
        title: String(m.how_title || m.key || "").trim(),
        text: String(m.how_text || "").trim(),
      }))
      .filter((x) => x.title && x.text);

    return [...base, ...HOW_META_ITEMS];
  }, [methods]);

  const metaSelected =
    selected.includes(META_KEY) ||
    selected.includes(META_FAST_KEY) ||
    selected.includes(META_ONLY_KEY) ||
    selected.includes(META_LEGACY_KEY);

  useEffect(() => {
    if (metaSelected && selected.length > 1) {
      const next = selected.includes(META_ONLY_KEY)
        ? [META_ONLY_KEY]
        : selected.includes(META_FAST_KEY)
        ? [META_FAST_KEY]
        : [META_KEY];
      setSelected(next);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metaSelected]);

  const sliderRef = useRef<HTMLDivElement | null>(null);
  const [valueLeftPx, setValueLeftPx] = useState(0);

  const THUMB_RADIUS = 8;

  function updateValueLeftPx(nextThreshold: number) {
    const el = sliderRef.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const usableWidth = Math.max(0, rect.width - THUMB_RADIUS * 2);

    const t = clamp01(nextThreshold);
    const x = THUMB_RADIUS + t * usableWidth;

    setValueLeftPx(x);
  }

  useEffect(() => {
    updateValueLeftPx(threshold);

    const el = sliderRef.current;
    if (!el) return;

    const ro = new ResizeObserver(() => updateValueLeftPx(threshold));
    ro.observe(el);

    const onResize = () => updateValueLeftPx(threshold);
    window.addEventListener("resize", onResize);

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold]);

  useEffect(() => {
    let pieUrlToRevoke: string | null = null;

    async function run() {
      if (!batchZip) {
        setBatchResultsCsv(null);
        setBatchSummaryJson(null);
        setBatchPiePng(null);
        setBatchCounts(null);
        setBatchPieUrl(null);
        return;
      }

      try {
        const u8 = await blobToU8(batchZip);
        const files = unzipSync(u8);

        const csvU8 = pickZipEntry(files, "results.csv");
        const summaryU8 = pickZipEntry(files, "summary.json");
        const pieU8 = pickZipEntry(files, "pie.png");
        const countsU8 = pickZipEntry(files, "counts.json");

        setBatchResultsCsv(
          csvU8
            ? new Blob([u8ToBlobPart(csvU8)], {
                type: "text/csv; charset=utf-8",
              })
            : null
        );
        setBatchSummaryJson(
          summaryU8
            ? new Blob([u8ToBlobPart(summaryU8)], {
                type: "application/json",
              })
            : null
        );

        if (pieU8) {
          const pieBlob = new Blob([u8ToBlobPart(pieU8)], { type: "image/png" });
          setBatchPiePng(pieBlob);
          const url = URL.createObjectURL(pieBlob);
          pieUrlToRevoke = url;
          setBatchPieUrl(url);
        } else {
          setBatchPiePng(null);
          setBatchPieUrl(null);
        }

        if (countsU8) {
          try {
            const txt = strFromU8(countsU8);
            const obj = JSON.parse(txt || "{}");
            const parsed: BatchCounts = {
              real: Number(obj?.real ?? 0),
              ai: Number(obj?.ai ?? 0),
              unknown: Number(obj?.unknown ?? 0),
            };
            setBatchCounts(parsed);
          } catch {
            setBatchCounts(null);
          }
        } else {
          setBatchCounts(null);
        }
      } catch (e: any) {
        setErr(humanizeErrorMessage(e));
        setBatchResultsCsv(null);
        setBatchSummaryJson(null);
        setBatchPiePng(null);
        setBatchCounts(null);
        setBatchPieUrl(null);
      }
    }

    run();

    return () => {
      if (pieUrlToRevoke) URL.revokeObjectURL(pieUrlToRevoke);
    };
  }, [batchZip]);

  async function analyze(fileToAnalyze?: File) {
    const f = fileToAnalyze ?? file;
    if (!f) return;

    if (selected.length === 0) {
      setMethodsError("Select at least one verification method.");
      return;
    }
    setMethodsError(null);

    setBusy(true);
    setErr(null);
    setReport(null);

    setBatchZip(null);
    setBatchResultsCsv(null);
    setBatchSummaryJson(null);
    setBatchPiePng(null);
    setBatchCounts(null);
    setBatchPieUrl(null);

    try {
      const fd = new FormData();
      fd.append("file", f);
      fd.append("threshold", String(threshold));

      const normalizedSelected = selected.map((x) =>
        x === META_LEGACY_KEY ? META_KEY : x
      );

      fd.append("methods_json", JSON.stringify(normalizedSelected));

      if (isZipFile(f)) {
        const blob = await postFormBlob("/analyze_zip", fd, {
          expectedContentType: "application/zip",
        });
        setBatchZip(blob);
        setReport(null);
      } else {
        const r = await postForm<Report>("/analyze", fd);
        setReport(r);
        setBatchZip(null);
      }
    } catch (e: any) {
      setErr(humanizeErrorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const components: any = report?.result.components ?? {};
  const scoreAi = report?.result.score_ai;

  const c2paScore = components["c2pa"];

  const verdict = useMemo(() => {
    if (!report) return null;

    if (isNullish(report.result?.score_ai)) {
      return { displayLabel: "UNKNOWN", showScoreBar: false };
    }

    const comps: any = report.result.components || {};

    const methodsRun = Object.keys(comps).filter(
      (k) =>
        k !== META_KEY &&
        k !== META_FAST_KEY &&
        k !== META_ONLY_KEY &&
        k !== META_LEGACY_KEY
    );

    let displayLabel = report.result.label;
    let showScoreBar = true;

    const c2pa = comps["c2pa"];
    const prnu = comps["prnu"];

    if (methodsRun.length === 1 && methodsRun[0] === "c2pa") {
      if (isNullish(c2pa)) {
        displayLabel = "UNKNOWN";
        showScoreBar = false;
      }
    }

    if (methodsRun.includes("prnu") && isNullish(prnu)) {
      const onlyPrnu = methodsRun.length === 1 && methodsRun[0] === "prnu";
      const prnuPlusC2pa = methodsRun.length === 2 && methodsRun.includes("c2pa");

      if (onlyPrnu || prnuPlusC2pa) {
        const c2paMissing = !methodsRun.includes("c2pa");
        const c2paUnknown = isNullish(c2pa);
        if (c2paMissing || c2paUnknown) {
          displayLabel = "UNKNOWN";
          showScoreBar = false;
        }
      }
    }

    return { displayLabel, showScoreBar };
  }, [report]);

  const visEntries = useMemo(() => {
    const v = report?.visuals ?? {};
    return Object.entries(v).sort(([a], [b]) => a.localeCompare(b));
  }, [report]);

  const metricsFiltered = useMemo(() => {
    const m = report?.metrics ?? {};
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(m)) {
      if (!k.startsWith("c2pa_")) out[k] = v;
    }
    return out;
  }, [report]);

  const displayComponents = useMemo(() => {
    const comps = { ...(report?.result.components || {}) } as Record<string, any>;
    if ("prnu" in comps && isNullish(comps.prnu)) comps.prnu = "unknown";
    if ("c2pa" in comps && isNullish(comps.c2pa)) comps.c2pa = "unknown";
    if (META_LEGACY_KEY in comps && !(META_KEY in comps)) {
      (comps as any)[META_KEY] = (comps as any)[META_LEGACY_KEY];
      delete (comps as any)[META_LEGACY_KEY];
    }
    return comps;
  }, [report]);

  const showBackendHint =
    err?.includes("NetworkError") || err?.includes("Failed to fetch");

  const attribGenerator = report?.attribution?.generator;
  const attribSdVariant = report?.attribution?.sd_variant;

  function toggleMeta(checked: boolean) {
    if (checked) setSelected([META_KEY]);
    else
      setSelected((prev) =>
        prev.filter((x) => x !== META_KEY && x !== META_LEGACY_KEY)
      );

    setMethodsError(null);
  }

  function toggleMetaFast(checked: boolean) {
    if (checked) setSelected([META_FAST_KEY]);
    else setSelected((prev) => prev.filter((x) => x !== META_FAST_KEY));

    setMethodsError(null);
  }

  function toggleMetaOnly(checked: boolean) {
    if (checked) setSelected([META_ONLY_KEY]);
    else setSelected((prev) => prev.filter((x) => x !== META_ONLY_KEY));

    setMethodsError(null);
  }

  function toggleBase(key: string, checked: boolean) {
    setSelected((prev) => {
      let next = prev.filter(
        (x) =>
          x !== META_KEY &&
          x !== META_FAST_KEY &&
          x !== META_ONLY_KEY &&
          x !== META_LEGACY_KEY
      );

      if (checked) next = Array.from(new Set([...next, key]));
      else next = next.filter((x) => x !== key);

      return next;
    });

    setMethodsError(null);
  }

  async function onPickFile(f: File | null) {
    if (busy) return;

    setFile(f);
    setReport(null);

    setBatchZip(null);
    setBatchResultsCsv(null);
    setBatchSummaryJson(null);
    setBatchPiePng(null);
    setBatchCounts(null);
    setBatchPieUrl(null);

    setErr(null);

    if (!f) return;

    if (selected.length === 0) {
      setMethodsError("Select at least one verification method.");
      return;
    }

    setMethodsError(null);
    await analyze(f);
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="flex">
        {/* SIDEBAR */}
        <aside className="w-[300px] bg-[#f7f9fb] border-r min-h-screen px-5 py-6 overflow-y-auto">
          <div className="space-y-6 text-sm">
            <div className="text-3xl font-semibold mb-8 leading-none">
              Settings
            </div>

            {/* Threshold */}
            <div className="space-y-2">
              <div className="flex items-center justify-between -mt-1">
                <div className="text-sm">AI verdict threshold</div>
                <Tooltip content="Decision threshold between 'AI' and 'REAL'. For meta-model it applies to meta score.">
                  <HelpCircle
                    className="h-4 w-4 text-muted-foreground cursor-help"
                    strokeWidth={2}
                  />
                </Tooltip>
              </div>

              <div className="relative pt-8" ref={sliderRef}>
                <div
                  className="absolute -top-1 text-xs font-medium tracking-tight text-[#ff4b4b]"
                  style={{
                    left: `${valueLeftPx}px`,
                    transform: "translateX(-50%)",
                  }}
                >
                  {threshold.toFixed(2)}
                </div>

                <Slider
                  value={[threshold]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={(v) => {
                    const next = v[0] ?? 0.5;
                    setThreshold(next);
                    updateValueLeftPx(next);
                  }}
                />
              </div>

              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>0.00</span>
                <span>1.00</span>
              </div>
            </div>

            <div className="border-t" />

            {/* Methods */}
            <div className="space-y-3">
              <div className="font-semibold">Verification methods</div>

              <div className="space-y-3">
                {methods
                  .filter(
                    (m) =>
                      m.key !== META_KEY &&
                      m.key !== META_FAST_KEY &&
                      m.key !== META_ONLY_KEY &&
                      m.key !== META_LEGACY_KEY
                  )
                  .sort((a, b) => a.key.localeCompare(b.key))
                  .map((m) => (
                    <div key={m.key} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        className="h-4 w-4 accent-[#ff4b4b] rounded-[3px]"
                        checked={selected.includes(m.key)}
                        onChange={(e) => toggleBase(m.key, e.target.checked)}
                      />
                      <div className="text-sm">{m.key}</div>
                      {m.description && (
                        <Tooltip content={m.description}>
                          <HelpCircle
                            className="h-4 w-4 text-muted-foreground cursor-help"
                            strokeWidth={2}
                          />
                        </Tooltip>
                      )}
                    </div>
                  ))}

                {/* Meta: Combined */}
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#ff4b4b] rounded-[3px]"
                    checked={
                      selected.includes(META_KEY) ||
                      selected.includes(META_LEGACY_KEY)
                    }
                    onChange={(e) => toggleMeta(e.target.checked)}
                  />
                  <div className="text-sm">{META_KEY}</div>
                  <Tooltip content="C2PA + PRNU gates, then meta-model over base detectors.">
                    <HelpCircle
                      className="h-4 w-4 text-muted-foreground cursor-help"
                      strokeWidth={2}
                    />
                  </Tooltip>
                </div>

                {/* Meta: Fast */}
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#ff4b4b] rounded-[3px]"
                    checked={selected.includes(META_FAST_KEY)}
                    onChange={(e) => toggleMetaFast(e.target.checked)}
                  />
                  <div className="text-sm">{META_FAST_KEY}</div>
                  <Tooltip content="C2PA gate, then meta-model (skips PRNU entirely).">
                    <HelpCircle
                      className="h-4 w-4 text-muted-foreground cursor-help"
                      strokeWidth={2}
                    />
                  </Tooltip>
                </div>

                {/* Meta: Meta-model only */}
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    className="h-4 w-4 accent-[#ff4b4b] rounded-[3px]"
                    checked={selected.includes(META_ONLY_KEY)}
                    onChange={(e) => toggleMetaOnly(e.target.checked)}
                  />
                  <div className="text-sm">{META_ONLY_KEY}</div>
                  <Tooltip content="Meta-model only: runs only required base detectors (no gates).">
                    <HelpCircle
                      className="h-4 w-4 text-muted-foreground cursor-help"
                      strokeWidth={2}
                    />
                  </Tooltip>
                </div>

                {methodsError && (
                  <div className="mt-2 text-xs text-red-600">{methodsError}</div>
                )}
              </div>
            </div>

            <div className="border-t" />

            {/* How does it work? */}
            <div className="space-y-2">
              <div className="font-semibold text-sm">How does it work?</div>
              <ul className="list-disc pl-5 space-y-4 text-[13px] leading-relaxed">
                {howItems.length ? (
                  howItems.map((it, idx) => (
                    <li key={`${it.title}-${idx}`}>
                      <b>{it.title}</b>: {it.text}
                    </li>
                  ))
                ) : (
                  <li className="text-muted-foreground">—</li>
                )}
              </ul>
            </div>
          </div>
        </aside>

        {/* MAIN */}
        <main className="flex-1 px-12 py-10">
          <div className="max-w-[1100px]">
            <h1 className="text-4xl font-semibold tracking-tight">
              AI-Image Detector
            </h1>
            <p className="text-sm text-muted-foreground mt-2">
              AI image authenticity analysis based on multiple independent signals.
              Not a forensic tool.
            </p>

            {/* Upload */}
            <div className="mt-6">
              <div className="text-sm font-medium mb-2">
                Upload an image (JPG/JPEG/PNG/BMP) or a ZIP with images
              </div>

              <div
                className={`flex items-center justify-between gap-4 rounded-md border bg-[#f3f6f9] px-4 py-4 ${
                  busy ? "opacity-90" : ""
                }`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (busy) return;
                  const f = e.dataTransfer.files?.[0] ?? null;
                  if (f) onPickFile(f);
                }}
              >
                <div className="flex items-center gap-4">
                  <div className="text-2xl text-muted-foreground">☁️</div>
                  <div>
                    <div className="text-sm font-medium">
                      {file ? file.name : "Drag and drop file here"}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Limit 200MB per file • ZIP: max 5000 images • max 60MB per image (uncompressed) • JPG, JPEG, PNG, BMP, ZIP
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <input
                    id="file"
                    type="file"
                    accept="image/png,image/jpeg,image/bmp,application/zip,.zip"
                    className="hidden"
                    disabled={busy}
                    onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
                  />

                  <label
                    htmlFor="file"
                    className={`inline-flex items-center rounded-md border bg-white px-3 py-2 text-sm cursor-pointer hover:bg-gray-50 ${
                      busy ? "pointer-events-none opacity-60" : ""
                    }`}
                  >
                    Browse files
                  </label>

                  {busy && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-transparent" />
                      <span>Analyzing… backend is processing</span>
                    </div>
                  )}
                </div>
              </div>

              {err && (
                <div className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {err}
                  {showBackendHint && (
                    <div className="mt-2 text-xs text-red-700/80">
                      Wygląda jak brak backendu lub CORS. Sprawdź czy API działa
                      na <b>{API_BASE}</b> i ma endpointy{" "}
                      <code>/methods</code>, <code>/analyze</code> oraz{" "}
                      <code>/analyze_zip</code>.
                    </div>
                  )}
                </div>
              )}
            </div>

            {(report || batchZip) && (
              <>
                <div className="mt-10 border-t pt-8" />

                {/* IMAGE REPORT UI */}
                {report && (
                  <>
                    {/* TOP ROW */}
                    <div className="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-10">
                      <div>
                        <div className="text-xl font-semibold mb-3">Original</div>
                        {previewUrl ? (
                          <img
                            src={previewUrl}
                            alt="original"
                            className="h-auto w-auto max-w-full"
                            style={{ maxWidth: MAX_DISPLAY_WIDTH }}
                          />
                        ) : (
                          <div className="text-sm text-muted-foreground">—</div>
                        )}
                      </div>

                      <div>
                        <div className="text-xl font-semibold mb-3">
                          EXIF (selected)
                        </div>
                        {Object.keys(report.exif || {}).length ? (
                          <pre className="max-h-[260px] overflow-auto rounded-md border bg-[#f7f9fb] p-3 text-xs">
                            {JSON.stringify(report.exif, null, 2)}
                          </pre>
                        ) : (
                          <div className="text-sm text-muted-foreground">
                            No key EXIF fields or metadata removed.
                          </div>
                        )}

                        {hasOwn(components, "c2pa") && (
                          <>
                            <div className="mt-6 border-t pt-6" />
                            <div className="text-xl font-semibold mb-3">
                              C2PA manifest
                            </div>

                            {isNullish(c2paScore) ? (
                              <div className="text-sm text-muted-foreground">
                                C2PA: no known Generative AI indicators were
                                detected (either the manifest is missing or
                                contains neutral information).
                              </div>
                            ) : (
                              <div className="text-sm">
                                <div>
                                  <b>C2PA score:</b> {String(c2paScore)}
                                </div>

                                {(report.metrics?.c2pa_label ||
                                  report.metrics?.c2pa_explanation) && (
                                  <div className="mt-2 space-y-1">
                                    {report.metrics?.c2pa_label ? (
                                      <div>
                                        <b>Label:</b>{" "}
                                        {String(report.metrics.c2pa_label)}
                                      </div>
                                    ) : null}
                                    {report.metrics?.c2pa_explanation ? (
                                      <div>
                                        {String(report.metrics.c2pa_explanation)}
                                      </div>
                                    ) : null}
                                  </div>
                                )}

                                {!report.metrics?.c2pa_label &&
                                  !report.metrics?.c2pa_explanation && (
                                    <div className="mt-2 text-sm text-muted-foreground">
                                      No C2PA explanation fields provided by
                                      backend.
                                    </div>
                                  )}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>

                    {/* Visualizations */}
                    <div className="mt-10 border-t pt-8" />
                    <div className="text-xl font-semibold mb-3">
                      Visualizations
                    </div>

                    {visEntries.length === 0 ? (
                      <div className="text-sm text-muted-foreground">
                        No visualizations returned by methods.
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {visEntries.map(([k, b64]) => (
                          <div key={k} className="space-y-2">
                            <div className="text-sm font-semibold">{k}</div>
                            <img
                              src={b64pngToDataUrl(b64 as any)}
                              alt={k}
                              className="w-full h-auto"
                              style={{ maxWidth: MAX_DISPLAY_WIDTH }}
                            />
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Result */}
                    <div className="mt-10 border-t pt-8" />
                    <div className="text-xl font-semibold mb-3">Result</div>

                    {verdict && (
                      <div className="space-y-3">
                        {verdict.showScoreBar && isFiniteNumber(scoreAi) ? (
                          <>
                            <div className="flex items-center justify-between">
                              <div className="text-sm font-medium">
                                Verdict{" "}
                                <span className="ml-2 font-semibold">
                                  {verdict.displayLabel}
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground">
                                score={scoreAi.toFixed(3)}
                              </div>
                            </div>
                            <Progress
                              value={clamp01(scoreAi) * 100}
                              className="h-2 rounded-md"
                            />
                          </>
                        ) : (
                          <div className="text-sm font-medium">
                            Verdict{" "}
                            <span className="ml-2 font-semibold">
                              {verdict.displayLabel}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Metric details */}
                    <div className="mt-6">
                      <details className="rounded-md border bg-white p-3">
                        <summary className="cursor-pointer text-sm font-medium">
                          Metric details
                        </summary>
                        <pre className="mt-3 max-h-[380px] overflow-auto rounded-md border bg-[#f7f9fb] p-3 text-xs">
                          {JSON.stringify(
                            {
                              ...metricsFiltered,
                              components: displayComponents,
                            },
                            null,
                            2
                          )}
                        </pre>
                      </details>
                    </div>

                    {/* Attribution */}
                    {(attribGenerator || attribSdVariant) && (
                      <>
                        <div className="mt-10 border-t pt-8" />
                        <div className="text-xl font-semibold mb-3">
                          Attribution
                        </div>

                        <div className="space-y-6">
                          <AttributionBlock
                            title="AI generator"
                            stage={attribGenerator}
                          />
                          <AttributionBlock
                            title="Stable Diffusion variant"
                            stage={attribSdVariant}
                          />
                        </div>
                      </>
                    )}
                  </>
                )}

                {/* Batch preview */}
                {batchZip && (
                  <>
                    <div className="text-xl font-semibold mb-3">
                      Batch summary
                    </div>

                    <div className="rounded-md border bg-white p-4">
                      {batchCounts ? (
                        <div className="text-sm text-muted-foreground">
                          REAL: <b>{batchCounts.real}</b> • AI:{" "}
                          <b>{batchCounts.ai}</b> • UNKNOWN:{" "}
                          <b>{batchCounts.unknown}</b>
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground">
                          Batch summary files received.
                        </div>
                      )}

                      {batchPieUrl && (
                        <div className="mt-4">
                          <img
                            src={batchPieUrl}
                            alt="Batch pie chart"
                            className="h-auto w-auto max-w-full"
                            style={{ maxWidth: 420 }}
                          />
                        </div>
                      )}
                    </div>
                  </>
                )}

                {/* Downloads */}
                <div className="mt-10 border-t pt-8" />
                <div className="text-xl font-semibold mb-3">Downloads</div>

                <div className="flex flex-wrap gap-3">
                  {/* Single-image downloads */}
                  {report && (
                    <>
                      <Button
                        className="rounded-md"
                        variant="outline"
                        onClick={() => {
                          try {
                            const light = makeLightReportFrontend(report);
                            const data = JSON.stringify(light, null, 2);
                            const blob = new Blob([data], {
                              type: "application/json",
                            });
                            downloadBlob(blob, "report.json");
                          } catch (e: any) {
                            setErr(humanizeErrorMessage(e));
                          }
                        }}
                      >
                        Download report (JSON)
                      </Button>

                      <Button
                        className="rounded-md"
                        variant="outline"
                        onClick={() => {
                          try {
                            const zipBlob = buildVisualsZipFrontend(report);
                            downloadBlob(zipBlob, "visuals.zip");
                          } catch (e: any) {
                            setErr(humanizeErrorMessage(e));
                          }
                        }}
                      >
                        Download visualizations (ZIP)
                      </Button>
                    </>
                  )}

                  {/* Batch downloads: 3 separate files */}
                  {batchResultsCsv && (
                    <Button
                      className="rounded-md"
                      variant="outline"
                      onClick={() => downloadBlob(batchResultsCsv, "results.csv")}
                    >
                      Download results (CSV)
                    </Button>
                  )}

                  {batchSummaryJson && (
                    <Button
                      className="rounded-md"
                      variant="outline"
                      onClick={() => downloadBlob(batchSummaryJson, "summary.json")}
                    >
                      Download summary (JSON)
                    </Button>
                  )}

                  {batchPiePng && (
                    <Button
                      className="rounded-md"
                      variant="outline"
                      onClick={() => downloadBlob(batchPiePng, "pie.png")}
                    >
                      Download chart (PNG)
                    </Button>
                  )}
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

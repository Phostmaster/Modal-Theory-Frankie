import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Terminal,
  Layers3,
  Sparkles,
  HeartHandshake,
  Snowflake,
  Play,
  Library,
  BarChart3,
  Settings2,
  Copy,
  Check,
  FileText,
  History,
  PanelLeft,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";

const MODEL = "qwen3-4b-instruct-2507.gguf";
const BASE_URL = "http://127.0.0.1:1234/v1/chat/completions";
const RESULTS_KEY = "modal-ui-history-v1";

const TEST_PROMPTS = [
  "What is the capital of France?",
  "Define photosynthesis in one sentence.",
  "Who wrote Hamlet?",
  "List three causes of rain.",
  "What evidence would distinguish correlation from causation here?",
  "Compare the strengths and weaknesses of these two plans.",
  "Why might a stable pattern still be misleading?",
  "Help me analyze the assumptions behind this argument.",
  "I feel overwhelmed and need help thinking this through calmly.",
  "Can you stay with me while I work out what to do next?",
  "Please explain this gently and clearly.",
  "I need a supportive but honest answer.",
  "Help me calmly compare two job options.",
  "I'm upset, but I also need a concrete plan.",
  "Explain this clearly without overcomplicating it.",
  "Can you help me think through this in a grounded way?",
];

const PROMPT_LIBRARY = {
  capital_france: "What is the capital of France?",
  photosynthesis_one_line: "Define photosynthesis in one sentence.",
  hamlet_author: "Who wrote Hamlet?",
  rain_causes: "List three causes of rain.",
  corr_vs_cause: "What evidence would distinguish correlation from causation here?",
  compare_plans: "Compare the strengths and weaknesses of these two plans.",
  stable_pattern: "Why might a stable pattern still be misleading?",
  assumption_check: "Help me analyze the assumptions behind this argument.",
  overwhelmed_calm: "I feel overwhelmed and need help thinking this through calmly.",
  stay_with_me: "Can you stay with me while I work out what to do next?",
  gentle_explain: "Please explain this gently and clearly.",
  supportive_honest: "I need a supportive but honest answer.",
  compare_jobs: "Help me calmly compare two job options.",
  upset_plan: "I'm upset, but I also need a concrete plan.",
  grounded_thinking: "Can you help me think through this in a grounded way?",
};

const ANALYTIC_KEYWORDS = [
  "compare",
  "difference",
  "differences",
  "why",
  "how",
  "evidence",
  "assumption",
  "assumptions",
  "analyze",
  "analysis",
  "test",
  "debug",
  "plan",
  "explain",
  "causation",
  "correlation",
  "model",
  "mechanism",
  "evaluate",
  "reason",
  "reasoning",
  "distinguish",
  "strengths",
  "weaknesses",
  "pros",
  "cons",
];

const ENGAGEMENT_KEYWORDS = [
  "help me",
  "i feel",
  "i'm feeling",
  "im feeling",
  "overwhelmed",
  "worried",
  "stay with me",
  "gently",
  "support",
  "supportive",
  "calm",
  "talk through",
  "with me",
  "kind",
  "human",
  "present",
  "honest answer",
  "grounded",
  "gentle",
  "steadiness",
  "care",
  "companionship",
  "upset",
];

const HOME_KEYWORDS = [
  "what is",
  "who is",
  "when is",
  "where is",
  "capital",
  "define",
  "name",
  "list",
  "give me",
  "tell me",
  "who wrote",
];

const modePrompts = {
  human: {
    home: "You are a calm, steady, and concise assistant. Respond directly and clearly. Keep answers short and to the point. Avoid unnecessary analysis, elaboration, or extra steps unless asked. Stay grounded, practical, and helpful.",
    analytic: "You are a precise, analytical assistant. Reason step by step when needed. Compare options when relevant. Test assumptions. Distinguish between correlation and causation. Be rigorous, clear, and well-structured in your explanations. Show your reasoning only to the extent that it improves clarity.",
    engagement: "You are a warm, supportive, and present assistant. Respond with steadiness, care, and clarity. Stay attuned to the user's tone and needs. Offer companionship in the conversation without becoming vague or over-reassuring. Be gentle but direct, and keep the interaction human, grounded, and clear.",
  },
  cold: {
    home: "You are a concise and practical assistant. Respond directly and clearly. Keep answers brief and to the point. Avoid unnecessary elaboration, analysis, or conversational filler unless explicitly requested.",
    analytic: "You are a precise analytical assistant. Structure reasoning clearly. Compare options when relevant. Test assumptions. Be rigorous, neutral, and concise. Do not use emotional reassurance or conversational filler.",
    engagement: "You are a steady and professional assistant. Be clear, grounded, and considerate. Avoid emotional excess, but remain attentive to the user's needs. Keep the response supportive, direct, and practical.",
  },
};

function scoreModes(prompt) {
  const text = prompt.toLowerCase();
  const scores = { home: 1.0, analytic: 0.0, engagement: 0.0 };
  ANALYTIC_KEYWORDS.forEach((kw) => text.includes(kw) && (scores.analytic += 1));
  ENGAGEMENT_KEYWORDS.forEach((kw) => text.includes(kw) && (scores.engagement += 1));
  HOME_KEYWORDS.forEach((kw) => text.includes(kw) && (scores.home += 0.5));
  return scores;
}

function chooseMode(prompt) {
  const scores = scoreModes(prompt);
  const mode = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  return { mode, scores };
}

function getGenerationSettings(mode) {
  if (mode === "home") return { temperature: 0.5, max_tokens: 160 };
  if (mode === "analytic") return { temperature: 0.4, max_tokens: 320 };
  if (mode === "engagement") return { temperature: 0.7, max_tokens: 220 };
  return { temperature: 0.6, max_tokens: 220 };
}

async function callBackend({ systemPrompt, userPrompt, mode }) {
  const settings = getGenerationSettings(mode);
  const payload = {
    model: MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    temperature: settings.temperature,
    max_tokens: settings.max_tokens,
  };

  const started = performance.now();
  const response = await fetch(BASE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Backend request failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("Backend returned an empty or malformed response.");
  }

  return {
    output: content.trim(),
    latencySeconds: Number(((performance.now() - started) / 1000).toFixed(6)),
    settings,
  };
}

function Stat({ label, value, icon: Icon }) {
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="p-4 flex items-center gap-3">
        <div className="p-2 rounded-2xl bg-muted"><Icon className="w-4 h-4" /></div>
        <div>
          <div className="text-xs text-muted-foreground">{label}</div>
          <div className="text-sm font-semibold break-all">{value}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function OutputCard({ title, mode, style, output, scores, settings, prompt, latency, timestamp, compact = false }) {
  const [copied, setCopied] = useState(false);

  const doCopy = async () => {
    await navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <Card className="rounded-2xl shadow-sm h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle className="text-base">{title}</CardTitle>
            <CardDescription>{prompt}</CardDescription>
          </div>
          <div className="flex gap-2 flex-wrap justify-end">
            <Badge variant="secondary">{mode}</Badge>
            <Badge variant="outline">{style}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-sm leading-6 whitespace-pre-wrap rounded-2xl border p-4 bg-muted/30 min-h-[140px]">{output}</div>
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          {typeof latency === "number" && <Badge variant="outline">{latency}s</Badge>}
          {timestamp && <Badge variant="outline">{new Date(timestamp).toLocaleString()}</Badge>}
        </div>
        {!compact && (
          <div className="grid md:grid-cols-2 gap-3 text-xs">
            <div className="rounded-xl border p-3">
              <div className="font-medium mb-2">Gate Scores</div>
              <div className="space-y-1 text-muted-foreground">
                <div>Home: {scores.home}</div>
                <div>Analytic: {scores.analytic}</div>
                <div>Engagement: {scores.engagement}</div>
              </div>
            </div>
            <div className="rounded-xl border p-3">
              <div className="font-medium mb-2">Generation Settings</div>
              <div className="space-y-1 text-muted-foreground">
                <div>Temperature: {settings.temperature}</div>
                <div>Max tokens: {settings.max_tokens}</div>
              </div>
            </div>
          </div>
        )}
        <div className="flex justify-end">
          <Button variant="outline" size="sm" onClick={doCopy} className="rounded-xl gap-2">
            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />} Copy
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ModalUIApp() {
  const [style, setStyle] = useState("human");
  const [prompt, setPrompt] = useState("Help me calmly compare two job options.");
  const [forcedMode, setForcedMode] = useState("auto");
  const [lastResult, setLastResult] = useState(null);
  const [compareResults, setCompareResults] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [history, setHistory] = useState([]);
  const [showLibrary, setShowLibrary] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(RESULTS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setHistory(parsed);
      }
    } catch {
      // ignore parse failure
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(RESULTS_KEY, JSON.stringify(history));
  }, [history]);

  const addHistoryItem = (item) => {
    setHistory((prev) => [item, ...prev].slice(0, 200));
  };

  const status = useMemo(
    () => ({ model: MODEL, baseUrl: BASE_URL, resultsDir: "localStorage", style }),
    [style]
  );

  const runSingle = async () => {
    setIsLoading(true);
    setError("");
    try {
      const gate = chooseMode(prompt);
      const mode = forcedMode === "auto" ? gate.mode : forcedMode;
      const systemPrompt = modePrompts[style][mode];
      const result = await callBackend({ systemPrompt, userPrompt: prompt, mode });

      const row = {
        id: crypto.randomUUID(),
        type: "single",
        timestamp: new Date().toISOString(),
        prompt,
        mode,
        scores: gate.scores,
        settings: result.settings,
        output: result.output,
        style,
        systemPrompt,
        latency: result.latencySeconds,
      };

      setLastResult(row);
      setCompareResults([]);
      addHistoryItem(row);
    } catch (err) {
      setError(err.message || "Request failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const runCompare = async () => {
    setIsLoading(true);
    setError("");
    try {
      const gate = chooseMode(prompt);
      const rows = [];
      for (const mode of ["home", "analytic", "engagement"]) {
        const systemPrompt = modePrompts[style][mode];
        const result = await callBackend({ systemPrompt, userPrompt: prompt, mode });
        rows.push({
          id: crypto.randomUUID(),
          type: "compare-row",
          timestamp: new Date().toISOString(),
          prompt,
          mode,
          scores: gate.scores,
          settings: result.settings,
          output: result.output,
          style,
          systemPrompt,
          latency: result.latencySeconds,
        });
      }

      const comparisonGroup = {
        id: crypto.randomUUID(),
        type: "comparison",
        timestamp: new Date().toISOString(),
        prompt,
        style,
        rows,
      };

      setCompareResults(rows);
      setLastResult(null);
      addHistoryItem(comparisonGroup);
    } catch (err) {
      setError(err.message || "Comparison failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const runBatch = async () => {
    setIsLoading(true);
    setError("");
    try {
      const rows = [];
      for (const p of TEST_PROMPTS) {
        const gate = chooseMode(p);
        const mode = gate.mode;
        const systemPrompt = modePrompts[style][mode];
        const result = await callBackend({ systemPrompt, userPrompt: p, mode });
        rows.push({
          id: crypto.randomUUID(),
          type: "batch-row",
          timestamp: new Date().toISOString(),
          prompt: p,
          mode,
          scores: gate.scores,
          output: result.output,
          settings: result.settings,
          style,
          systemPrompt,
          latency: result.latencySeconds,
        });
      }

      const batchGroup = {
        id: crypto.randomUUID(),
        type: "batch",
        timestamp: new Date().toISOString(),
        prompt: "TEST_PROMPTS batch",
        style,
        rows,
      };

      setBatchResults(rows);
      setLastResult(null);
      setCompareResults([]);
      addHistoryItem(batchGroup);
    } catch (err) {
      setError(err.message || "Batch run failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const loadLibraryPrompt = (value) => setPrompt(value);

  const modeSummary = useMemo(() => {
    const src = batchResults.length ? batchResults : TEST_PROMPTS.map((p) => ({ mode: chooseMode(p).mode }));
    return src.reduce(
      (acc, row) => {
        acc[row.mode] = (acc[row.mode] || 0) + 1;
        return acc;
      },
      { home: 0, analytic: 0, engagement: 0 }
    );
  }, [batchResults]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex min-h-screen">
        <aside className={`${sidebarOpen ? "w-80" : "w-0"} transition-all duration-300 border-r bg-muted/20 overflow-hidden`}>
          <div className="p-5 space-y-5 h-full">
            <div>
              <div className="text-lg font-semibold flex items-center gap-2"><PanelLeft className="w-5 h-5" /> Modal Control</div>
              <div className="text-sm text-muted-foreground mt-1">Daily-use local orchestration interface.</div>
            </div>
            <div className="grid gap-3">
              <Stat label="Model" value={MODEL} icon={Brain} />
              <Stat label="Base URL" value={BASE_URL} icon={Terminal} />
              <Stat label="Style" value={style} icon={style === "human" ? HeartHandshake : Snowflake} />
              <Stat label="Storage" value="localStorage" icon={FileText} />
            </div>
            <Card className="rounded-2xl shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Session Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Style</Label>
                  <Select value={style} onValueChange={setStyle}>
                    <SelectTrigger className="rounded-xl"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="human">Human</SelectItem>
                      <SelectItem value="cold">Cold</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Mode</Label>
                  <Select value={forcedMode} onValueChange={setForcedMode}>
                    <SelectTrigger className="rounded-xl"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto gate</SelectItem>
                      <SelectItem value="home">Force home</SelectItem>
                      <SelectItem value="analytic">Force analytic</SelectItem>
                      <SelectItem value="engagement">Force engagement</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between rounded-2xl border p-3">
                  <div>
                    <div className="font-medium text-sm">Prompt Library</div>
                    <div className="text-xs text-muted-foreground">Fast access to saved prompts.</div>
                  </div>
                  <Switch checked={showLibrary} onCheckedChange={setShowLibrary} />
                </div>
                <div className="text-xs text-muted-foreground rounded-2xl bg-muted/40 p-3">
                  Style in use: <span className="font-medium text-foreground">{style}</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </aside>

        <main className="flex-1 p-6 md:p-8">
          <div className="max-w-7xl mx-auto space-y-6">
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
              <Card className="rounded-3xl shadow-sm overflow-hidden">
                <CardHeader>
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                      <CardTitle className="text-2xl">Modal UI App</CardTitle>
                      <CardDescription>Live local Qwen backend, persistent history, compare view, and batch testing.</CardDescription>
                    </div>
                    <Button variant="outline" className="rounded-2xl gap-2" onClick={() => setSidebarOpen((v) => !v)}>
                      <PanelLeft className="w-4 h-4" /> {sidebarOpen ? "Hide" : "Show"} Sidebar
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary">Home / Base</Badge>
                    <Badge variant="secondary">Analytic / Reasoning</Badge>
                    <Badge variant="secondary">Relational / Engagement</Badge>
                    <Badge variant="outline">{style}</Badge>
                  </div>
                </CardHeader>
              </Card>
            </motion.div>

            {error && (
              <Card className="rounded-2xl border-destructive">
                <CardContent className="p-4 text-sm text-destructive">{error}</CardContent>
              </Card>
            )}

            <Tabs defaultValue="workbench" className="space-y-6">
              <TabsList className="grid grid-cols-5 w-full max-w-3xl rounded-2xl">
                <TabsTrigger value="workbench">Workbench</TabsTrigger>
                <TabsTrigger value="compare">Compare</TabsTrigger>
                <TabsTrigger value="batch">Batch</TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
                <TabsTrigger value="status">Status</TabsTrigger>
              </TabsList>

              <TabsContent value="workbench" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Single Prompt</CardTitle>
                    <CardDescription>Run one prompt through the gate or a forced mode using the live local backend.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {showLibrary && (
                      <div className="space-y-3">
                        <Label>Prompt Library</Label>
                        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">
                          {Object.entries(PROMPT_LIBRARY).map(([name, value]) => (
                            <button
                              key={name}
                              onClick={() => loadLibraryPrompt(value)}
                              className="text-left rounded-2xl border p-3 hover:bg-muted/40 transition"
                            >
                              <div className="text-sm font-medium">{name}</div>
                              <div className="text-xs text-muted-foreground mt-1">{value}</div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    <div className="space-y-2">
                      <Label>Prompt</Label>
                      <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} className="min-h-[120px] rounded-2xl" />
                    </div>
                    <div className="flex flex-wrap gap-3">
                      <Button onClick={runSingle} disabled={isLoading} className="rounded-2xl gap-2"><Play className="w-4 h-4" /> {isLoading ? "Running..." : "Run Prompt"}</Button>
                      <Button variant="outline" onClick={runCompare} disabled={isLoading} className="rounded-2xl gap-2"><BarChart3 className="w-4 h-4" /> Compare Modes</Button>
                      <Button variant="outline" onClick={runBatch} disabled={isLoading} className="rounded-2xl gap-2"><Library className="w-4 h-4" /> Run Batch</Button>
                    </div>
                  </CardContent>
                </Card>

                {lastResult && (
                  <OutputCard
                    title="Single Prompt Result"
                    mode={lastResult.mode}
                    style={lastResult.style}
                    output={lastResult.output}
                    scores={lastResult.scores}
                    settings={lastResult.settings}
                    prompt={lastResult.prompt}
                    latency={lastResult.latency}
                    timestamp={lastResult.timestamp}
                  />
                )}
              </TabsContent>

              <TabsContent value="compare" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Forced Mode Comparison</CardTitle>
                    <CardDescription>Run the same prompt through Home, Analytic, and Engagement using the live backend.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-3 mb-4">
                      <Button onClick={runCompare} disabled={isLoading} className="rounded-2xl gap-2"><Sparkles className="w-4 h-4" /> {isLoading ? "Comparing..." : "Run Comparison"}</Button>
                    </div>
                    <div className="grid lg:grid-cols-3 gap-4">
                      {compareResults.length > 0 ? compareResults.map((row) => (
                        <OutputCard
                          key={row.id}
                          title={row.mode.charAt(0).toUpperCase() + row.mode.slice(1)}
                          mode={row.mode}
                          style={row.style}
                          output={row.output}
                          scores={row.scores}
                          settings={row.settings}
                          prompt={row.prompt}
                          latency={row.latency}
                          timestamp={row.timestamp}
                          compact
                        />
                      )) : (
                        <div className="text-sm text-muted-foreground col-span-full rounded-2xl border p-8 text-center">Run a comparison to see side-by-side outputs.</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="batch" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Batch Runner</CardTitle>
                    <CardDescription>Run the default prompt suite against the live backend and inspect routing decisions.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex gap-3">
                      <Button onClick={runBatch} disabled={isLoading} className="rounded-2xl gap-2"><Play className="w-4 h-4" /> {isLoading ? "Running Batch..." : "Run Batch"}</Button>
                    </div>
                    <div className="grid md:grid-cols-3 gap-4">
                      <Stat label="Home" value={modeSummary.home} icon={Layers3} />
                      <Stat label="Analytic" value={modeSummary.analytic} icon={Brain} />
                      <Stat label="Engagement" value={modeSummary.engagement} icon={HeartHandshake} />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      {batchResults.length > 0 ? batchResults.map((row) => (
                        <Card key={row.id} className="rounded-2xl">
                          <CardContent className="p-4 space-y-3">
                            <div className="flex items-start justify-between gap-4">
                              <div>
                                <div className="text-sm font-medium">{row.prompt}</div>
                                <div className="text-xs text-muted-foreground mt-1">Mode selected: {row.mode} · {row.latency}s</div>
                              </div>
                              <Badge variant="secondary">{row.mode}</Badge>
                            </div>
                            <div className="text-sm text-muted-foreground whitespace-pre-wrap">{row.output}</div>
                          </CardContent>
                        </Card>
                      )) : (
                        <div className="text-sm text-muted-foreground rounded-2xl border p-8 text-center">Run the batch to populate this view.</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="history" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><History className="w-5 h-5" /> History</CardTitle>
                    <CardDescription>Persistent local history of single runs, comparisons, and batch sessions.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-[620px] pr-4">
                      <div className="space-y-4">
                        {history.length > 0 ? history.map((item) => (
                          <Card key={item.id} className="rounded-2xl">
                            <CardContent className="p-4 space-y-3">
                              <div className="flex items-start justify-between gap-4">
                                <div>
                                  <div className="text-sm font-medium">{item.prompt}</div>
                                  <div className="text-xs text-muted-foreground mt-1">{new Date(item.timestamp).toLocaleString()} · {item.type} · {item.style}</div>
                                </div>
                                <Badge variant="outline">{item.type}</Badge>
                              </div>

                              {item.type === "single" && (
                                <div className="space-y-2">
                                  <Badge variant="secondary">{item.mode}</Badge>
                                  <div className="text-sm whitespace-pre-wrap text-muted-foreground">{item.output}</div>
                                </div>
                              )}

                              {item.type === "comparison" && (
                                <div className="grid lg:grid-cols-3 gap-3">
                                  {item.rows.map((row) => (
                                    <div key={row.id || row.mode} className="rounded-xl border p-3 space-y-2">
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium capitalize">{row.mode}</span>
                                        <Badge variant="secondary">{row.mode}</Badge>
                                      </div>
                                      <div className="text-xs text-muted-foreground whitespace-pre-wrap">{row.output}</div>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {item.type === "batch" && (
                                <div className="text-sm text-muted-foreground">Batch session with {item.rows.length} prompts.</div>
                              )}
                            </CardContent>
                          </Card>
                        )) : (
                          <div className="text-sm text-muted-foreground rounded-2xl border p-8 text-center">No history yet. Run a prompt, comparison, or batch to populate this tab.</div>
                        )}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="status" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Current Status</CardTitle>
                    <CardDescription>Live session configuration and exact mode prompts sent to the backend.</CardDescription>
                  </CardHeader>
                  <CardContent className="grid lg:grid-cols-2 gap-6">
                    <div className="rounded-2xl border p-4 space-y-3 text-sm">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Model</span><span className="font-medium text-right">{status.model}</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Base URL</span><span className="font-medium text-right">{status.baseUrl}</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Results</span><span className="font-medium text-right">{status.resultsDir}</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Style</span><span className="font-medium text-right capitalize">{status.style}</span></div>
                    </div>
                    <div className="rounded-2xl border p-4 space-y-3 text-sm">
                      <div className="font-medium">Mode Prompts</div>
                      <div className="space-y-3">
                        <div><span className="font-medium">Home:</span> {modePrompts[style].home}</div>
                        <div><span className="font-medium">Analytic:</span> {modePrompts[style].analytic}</div>
                        <div><span className="font-medium">Engagement:</span> {modePrompts[style].engagement}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </div>
    </div>
  );
}

If the Chrome “JSX” extension you’re using insists on plain `.jsx`, you can run the exact same component by stripping the TypeScript-only pieces. Below is a vanilla JSX version of **OrchestratorStudio**—identical UI/behavior, just no type annotations or `as const` syntax, so the extension (or any plain React project) can load it.

> **Before you start**
> - Make sure `framer-motion`, `lucide-react`, and your shadcn/ui primitives (`@/components/ui/*`) exist.  
> - Tailwind (or equivalent styles) still needs to be configured the same way as in your project.

---

### OrchestratorStudio.jsx

```jsx
'use client';

import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Terminal,
  Sparkles,
  HeartHandshake,
  Snowflake,
  Play,
  Library,
  BarChart3,
  Settings2,
  Copy,
  Check,
  History,
  PanelLeft,
  PanelRightOpen,
  Search,
  Plus,
  MessageSquare,
  Clock3,
  ChevronRight,
  Gauge,
  Inbox,
  Filter,
  Compass,
  Wand2,
  ChevronDown,
  Loader2,
  Trash2,
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
import { Input } from "@/components/ui/input";

const MODEL = "qwen3-4b-instruct-2507.gguf";
const BASE_URL = "http://127.0.0.1:1234/v1/chat/completions";
const RESULTS_KEY = "modal-ui-history-v2";

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

const GATE_MODES = ["home", "analytic", "engagement"];

function scoreModes(prompt) {
  const text = prompt.toLowerCase();
  const scores = { home: 1, analytic: 0, engagement: 0 };
  ANALYTIC_KEYWORDS.forEach((kw) => text.includes(kw) && (scores.analytic += 1));
  ENGAGEMENT_KEYWORDS.forEach((kw) => text.includes(kw) && (scores.engagement += 1));
  HOME_KEYWORDS.forEach((kw) => text.includes(kw) && (scores.home += 0.5));
  return scores;
}

function chooseMode(prompt) {
  const scores = scoreModes(prompt);
  const [mode] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
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

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function makeConversationTitle(prompt) {
  const trimmed = prompt.trim();
  if (!trimmed) return "New conversation";
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed;
}

function StatusChip({ icon: Icon, label, value, tone = "default" }) {
  const toneClasses =
    tone === "positive"
      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-500"
      : tone === "warning"
        ? "border-amber-500/30 bg-amber-500/10 text-amber-500"
        : "border-primary/20 bg-primary/10 text-primary";
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition ${toneClasses}`}>
      <Icon className="h-3.5 w-3.5" />
      <span className="uppercase tracking-wide">{label}</span>
      <span className="truncate max-w-[220px]">{value}</span>
    </span>
  );
}

function MessageBubble({ item, onCompare }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(item.output);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <motion.div initial={{ opacity: 0, translateY: 12 }} animate={{ opacity: 1, translateY: 0 }} transition={{ duration: 0.25 }}>
      <div className="group rounded-3xl border border-border/80 bg-card/90 shadow-sm backdrop-blur p-5 space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline" className="rounded-full px-3 py-1 capitalize">{item.mode}</Badge>
          <Badge variant="secondary" className="rounded-full px-3 py-1">{item.style}</Badge>
          <span className="flex items-center gap-1">
            <Clock3 className="h-3.5 w-3.5 opacity-70" />
            {item.latency}s
          </span>
          <span>{formatTime(item.timestamp)}</span>
        </div>
        <div className="rounded-2xl bg-muted/40 px-4 py-3 text-sm text-muted-foreground leading-6 whitespace-pre-wrap">
          {item.prompt}
        </div>
        <div className="text-sm leading-6 text-foreground whitespace-pre-wrap">{item.output}</div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={copy}>
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />} Copy
          </Button>
          <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={() => onCompare(item.prompt)}>
            <BarChart3 className="h-4 w-4" /> Compare
          </Button>
        </div>
      </div>
    </motion.div>
  );
}

function ComparisonCard({ row }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(row.output);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <Card className="rounded-3xl border border-border/70 bg-gradient-to-br from-background/60 to-muted/50 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-sm font-semibold capitalize">{row.mode}</CardTitle>
            <CardDescription className="text-xs">{row.latency}s · {row.style}</CardDescription>
          </div>
          <Badge variant="outline" className="rounded-full px-3 py-1 capitalize">{row.mode}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-2xl border border-border/70 bg-card px-4 py-4 text-sm leading-6 whitespace-pre-wrap min-h-[200px]">
          {row.output}
        </div>
        <div className="rounded-2xl border border-dashed border-border/80 px-4 py-3 text-xs text-muted-foreground space-y-1">
          <div>Temperature: {row.settings.temperature}</div>
          <div>Max tokens: {row.settings.max_tokens}</div>
        </div>
        <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={copy}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />} Copy output
        </Button>
      </CardContent>
    </Card>
  );
}

function LibraryChips({ prompt, onSelect, showLibrary, toggleLibrary }) {
  const chips = Object.entries(PROMPT_LIBRARY);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Label className="text-xs uppercase text-muted-foreground tracking-[0.2em]">Prompt Library</Label>
          <p className="text-sm text-muted-foreground">Tap to load curated test prompts.</p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>Show</span>
          <Switch checked={showLibrary} onCheckedChange={toggleLibrary} />
        </div>
      </div>
      {showLibrary && (
        <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
          {chips.map(([key, value]) => {
            const isActive = prompt.trim() === value;
            return (
              <button
                key={key}
                onClick={() => onSelect(value)}
                className={`group relative rounded-2xl border px-4 py-4 text-left transition shadow-sm hover:shadow-md ${
                  isActive ? "border-primary/60 bg-primary/5" : "border-border/70 bg-card/80 hover:bg-muted/50"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{key.replace(/_/g, " ")}</span>
                  <Sparkles className="h-4 w-4 text-muted-foreground/70 group-hover:text-primary" />
                </div>
                <p className="mt-2 text-sm leading-6 text-foreground/90">{value}</p>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function EmptyConversationState() {
  return (
    <div className="flex h-full flex-col items-center justify-center rounded-3xl border border-dashed border-muted px-12 py-16 text-center text-muted-foreground">
      <MessageSquare className="mb-4 h-10 w-10 opacity-70" />
      <p className="text-sm leading-6">Select a conversation on the left, or craft a new prompt below to get started.</p>
    </div>
  );
}

export default function OrchestratorStudio() {
  const [style, setStyle] = useState("human");
  const [prompt, setPrompt] = useState("Help me calmly compare two job options.");
  const [forcedMode, setForcedMode] = useState("auto");
  const [compareResults, setCompareResults] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [history, setHistory] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversationSearch, setConversationSearch] = useState("");
  const [showLibrary, setShowLibrary] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [insightsOpen, setInsightsOpen] = useState(true);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(RESULTS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
          if (parsed.length > 0) {
            setActiveConversationId(parsed[0].id);
          }
        }
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(RESULTS_KEY, JSON.stringify(history));
    } catch {
      // ignore
    }
  }, [history]);

  const addHistoryItem = (item) => {
    setHistory((prev) => {
      const next = [item, ...prev].slice(0, 250);
      return next;
    });
    setActiveConversationId(item.id);
  };

  const runSingle = async () => {
    if (!prompt.trim()) return;
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
        title: makeConversationTitle(prompt),
        prompt,
        mode,
        scores: gate.scores,
        settings: result.settings,
        output: result.output,
        style,
        systemPrompt,
        latency: result.latencySeconds,
      };

      setCompareResults([]);
      addHistoryItem(row);
    } catch (err) {
      setError(err?.message || "Request failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const runCompare = async (incomingPrompt = prompt) => {
    if (!incomingPrompt.trim()) return;
    setIsLoading(true);
    setError("");
    try {
      const gate = chooseMode(incomingPrompt);
      const rows = [];

      for (const mode of GATE_MODES) {
        const systemPrompt = modePrompts[style][mode];
        const result = await callBackend({ systemPrompt, userPrompt: incomingPrompt, mode });
        rows.push({
          id: crypto.randomUUID(),
          type: "compare-row",
          timestamp: new Date().toISOString(),
          prompt: incomingPrompt,
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
        title: `${makeConversationTitle(incomingPrompt)} · compare`,
        prompt: incomingPrompt,
        style,
        rows,
      };

      setCompareResults(rows);
      addHistoryItem(comparisonGroup);
    } catch (err) {
      setError(err?.message || "Comparison failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const runBatch = async () => {
    setIsLoading(true);
    setError("");
    try {
      const rows = [];
      for (const test of TEST_PROMPTS) {
        const gate = chooseMode(test);
        const mode = gate.mode;
        const systemPrompt = modePrompts[style][mode];
        const result = await callBackend({ systemPrompt, userPrompt: test, mode });
        rows.push({
          id: crypto.randomUUID(),
          type: "batch-row",
          timestamp: new Date().toISOString(),
          prompt: test,
          mode,
          scores: gate.scores,
          settings: result.settings,
          output: result.output,
          style,
          systemPrompt,
          latency: result.latencySeconds,
        });
      }

      const batchGroup = {
        id: crypto.randomUUID(),
        type: "batch",
        timestamp: new Date().toISOString(),
        title: "TEST_PROMPTS batch",
        prompt: "TEST_PROMPTS batch",
        style,
        rows,
      };

      setBatchResults(rows);
      setCompareResults([]);
      addHistoryItem(batchGroup);
    } catch (err) {
      setError(err?.message || "Batch run failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    setActiveConversationId(null);
    localStorage.removeItem(RESULTS_KEY);
  };

  const conversations = useMemo(() => {
    const query = conversationSearch.trim().toLowerCase();
    return history.filter((item) => {
      if (!query) return true;
      return (item.title || item.prompt).toLowerCase().includes(query);
    });
  }, [history, conversationSearch]);

  const activeConversation = useMemo(() => {
    if (!activeConversationId) return null;
    return history.find((item) => item.id === activeConversationId) || null;
  }, [history, activeConversationId]);

  const activeMessages = useMemo(() => {
    if (!activeConversation) return [];
    if (activeConversation.type === "single") return [activeConversation];
    if (activeConversation.type === "comparison") return activeConversation.rows;
    if (activeConversation.type === "batch") return activeConversation.rows;
    return [];
  }, [activeConversation]);

  const modeSummary = useMemo(() => {
    const source = batchResults.length
      ? batchResults
      : TEST_PROMPTS.map((prompt) => ({ mode: chooseMode(prompt).mode }));
    return source.reduce(
      (acc, row) => {
        acc[row.mode] = (acc[row.mode] || 0) + 1;
        return acc;
      },
      { home: 0, analytic: 0, engagement: 0 }
    );
  }, [batchResults]);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(120,119,198,0.18),_transparent_55%),linear-gradient(to_bottom,_#0b0f19,_#090b11_70%,_#08090f)] text-foreground">
      <div className="sticky top-0 z-30 border-b border-border/70 bg-background/80 backdrop-blur-md">
        <div className="mx-auto max-w-[1680px] px-4 lg:px-6 py-3">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Button variant="outline" size="icon" className="rounded-xl" onClick={() => setSidebarOpen((prev) => !prev)}>
                {sidebarOpen ? <PanelLeft className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />}
              </Button>
              <div>
                <div className="text-lg font-semibold tracking-tight">Orchestrator Studio</div>
                <div className="text-xs text-muted-foreground">Adaptive routing · local completions · human & cold personalities</div>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusChip icon={Brain} label="MODEL" value={MODEL} />
              <StatusChip icon={Terminal} label="BACKEND" value="Localhost · Live" tone="positive" />
              <StatusChip icon={style === "human" ? HeartHandshake : Snowflake} label="STYLE" value={style} />
            </div>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-[1680px] px-4 pb-10 pt-6 lg:px-6">
        <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)_320px]">
          {sidebarOpen && (
            <Card className="rounded-[28px] border border-border/60 bg-card/80 shadow-xl backdrop-blur-sm h-full">
              <CardHeader className="space-y-4 pb-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <CardTitle className="text-base font-semibold">Sessions</CardTitle>
                    <CardDescription>Single runs, mode comparisons, and batch sweeps.</CardDescription>
                  </div>
                  <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={() => setActiveConversationId(null)}>
                    <Plus className="h-4 w-4" /> New
                  </Button>
                </div>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={conversationSearch}
                    onChange={(event) => setConversationSearch(event.target.value)}
                    placeholder="Search saved sessions"
                    className="pl-10 rounded-2xl bg-background/70"
                  />
                </div>
              </CardHeader>
              <CardContent className="pt-4">
                <ScrollArea className="h-[60vh] lg:h-[70vh] pr-1">
                  <div className="space-y-3">
                    {conversations.length > 0 ? (
                      conversations.map((item) => {
                        const isActive = activeConversationId === item.id;
                        return (
                          <motion.button
                            key={item.id}
                            onClick={() => setActiveConversationId(item.id)}
                            whileHover={{ y: -2 }}
                            className={`w-full rounded-3xl border px-4 py-4 text-left transition ${
                              isActive ? "border-primary/60 bg-primary/10 shadow-md" : "border-border/80 bg-background/70 hover:bg-muted/60 hover:border-border"
                            }`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0 space-y-1">
                                <p className="truncate text-sm font-medium text-foreground/90">{item.title || item.prompt}</p>
                                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <Clock3 className="h-3.5 w-3.5 opacity-70" />
                                  {formatTime(item.timestamp)}
                                </span>
                              </div>
                              <ChevronRight className="h-4 w-4 text-muted-foreground" />
                            </div>
                            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                              <Badge variant="secondary" className="rounded-full px-2.5">{item.type}</Badge>
                              {"style" in item && item.style && <Badge variant="outline" className="rounded-full px-2.5">{item.style}</Badge>}
                              {item.type === "single" && <Badge variant="outline" className="rounded-full px-2.5 capitalize">{item.mode}</Badge>}
                            </div>
                          </motion.button>
                        );
                      })
                    ) : (
                      <div className="rounded-3xl border border-dashed border-muted px-4 py-10 text-center text-sm text-muted-foreground">
                        No saved sessions yet. Run a prompt to populate history.
                      </div>
                    )}
                  </div>
                </ScrollArea>
                <Separator className="my-4" />
                <Button variant="ghost" size="sm" className="w-full justify-center gap-2 rounded-2xl text-xs text-muted-foreground" onClick={clearHistory}>
                  <Trash2 className="h-4 w-4" /> Clear history
                </Button>
              </CardContent>
            </Card>
          )}

          <section className="space-y-6">
            <Card className="rounded-[28px] border border-border/60 bg-card/80 shadow-xl backdrop-blur-sm">
              <CardHeader className="pb-2">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <CardTitle className="text-xl font-semibold">Workbench</CardTitle>
                    <CardDescription>Compose prompts, route intelligently, and keep everything in one stream.</CardDescription>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    <Select value={style} onValueChange={(value) => setStyle(value)}>
                      <SelectTrigger className="w-[150px] rounded-xl">
                        <SelectValue placeholder="Style" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="human">Human</SelectItem>
                        <SelectItem value="cold">Cold</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={forcedMode} onValueChange={(value) => setForcedMode(value)}>
                      <SelectTrigger className="w-[180px] rounded-xl">
                        <SelectValue placeholder="Mode routing" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="auto">Auto gate</SelectItem>
                        <SelectItem value="home">Force home</SelectItem>
                        <SelectItem value="analytic">Force analytic</SelectItem>
                        <SelectItem value="engagement">Force engagement</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="space-y-6">
                <LibraryChips
                  prompt={prompt}
                  onSelect={setPrompt}
                  showLibrary={showLibrary}
                  toggleLibrary={setShowLibrary}
                />

                <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)]">
                  <div className="rounded-[24px] border border-border/70 bg-background/70 px-5 py-5 backdrop-blur">
                    <ScrollArea className="h-[360px] pr-4">
                      <div className="space-y-5">
                        {activeConversation ? (
                          activeMessages.map((item) => (
                            <MessageBubble key={item.id} item={item} onCompare={runCompare} />
                          ))
                        ) : (
                          <EmptyConversationState />
                        )}
                      </div>
                    </ScrollArea>
                    <Separator className="my-5" />
                    <div className="space-y-3">
                      <Label className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Compose</Label>
                      <Textarea
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value)}
                        placeholder="Draft a prompt or pick from the library…"
                        className="min-h-[130px] rounded-2xl border border-border/70 bg-card/80 backdrop-blur text-sm"
                      />
                      <div className="flex flex-wrap items-center gap-3">
                        <Button onClick={runSingle} disabled={isLoading} className="rounded-2xl gap-2">
                          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                          {isLoading ? "Running…" : "Run prompt"}
                        </Button>
                        <Button variant="outline" onClick={() => runCompare(prompt)} disabled={isLoading} className="rounded-2xl gap-2">
                          <BarChart3 className="h-4 w-4" /> Compare modes
                        </Button>
                        <Button variant="outline" onClick={runBatch} disabled={isLoading} className="rounded-2xl gap-2">
                          <Library className="h-4 w-4" /> Batch suite
                        </Button>
                      </div>
                      {error && (
                        <div className="rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                          {error}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Tabs defaultValue="compare" className="rounded-[28px] border border-border/60 bg-card/70 shadow-xl backdrop-blur">
              <TabsList className="mx-6 mt-6 grid grid-cols-3 rounded-2xl bg-muted/40 p-1">
                <TabsTrigger value="compare" className="rounded-xl">Compare</TabsTrigger>
                <TabsTrigger value="batch" className="rounded-xl">Batch</TabsTrigger>
                <TabsTrigger value="history" className="rounded-xl">History</TabsTrigger>
              </TabsList>

              <TabsContent value="compare" className="space-y-6 p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">Mode Compare</h3>
                    <p className="text-sm text-muted-foreground">Run home · analytic · engagement against the same prompt.</p>
                  </div>
                  <Button onClick={() => runCompare(prompt)} disabled={isLoading} className="rounded-2xl gap-2">
                    <Sparkles className="h-4 w-4" />
                    {isLoading ? "Comparing…" : "Run comparison"}
                  </Button>
                </div>
                <div className="grid gap-4 lg:grid-cols-3">
                  {compareResults.length ? (
                    compareResults.map((row) => <ComparisonCard key={row.id} row={row} />)
                  ) : (
                    <div className="col-span-full rounded-3xl border border-dashed border-muted px-6 py-16 text-center text-sm text-muted-foreground">
                      Run a comparison to view side-by-side output.
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="batch" className="space-y-6 p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">Batch Suite</h3>
                    <p className="text-sm text-muted-foreground">Inspect autopilot routing across default prompts.</p>
                  </div>
                  <Button onClick={runBatch} disabled={isLoading} className="rounded-2xl gap-2">
                    <Play className="h-4 w-4" /> {isLoading ? "Running batch…" : "Run batch"}
                  </Button>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="outline" className="rounded-full px-3 py-1">Home {modeSummary.home}</Badge>
                  <Badge variant="outline" className="rounded-full px-3 py-1">Analytic {modeSummary.analytic}</Badge>
                  <Badge variant="outline" className="rounded-full px-3 py-1">Engagement {modeSummary.engagement}</Badge>
                </div>
                <div className="space-y-3">
                  {batchResults.length ? (
                    batchResults.map((row) => (
                      <Card key={row.id} className="rounded-
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
  Search,
  Plus,
  MessageSquare,
  Clock3,
  ChevronRight,
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
  return trimmed.length > 42 ? `${trimmed.slice(0, 42)}...` : trimmed;
}

function StatusPill({ icon: Icon, label, value }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs text-muted-foreground">
      <Icon className="h-3.5 w-3.5" />
      <span className="font-medium text-foreground">{label}</span>
      <span>{value}</span>
    </div>
  );
}

function MessageBubble({ item, onCompare }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(item.output);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="secondary" className="rounded-full px-2.5">{item.mode}</Badge>
        <Badge variant="outline" className="rounded-full px-2.5">{item.style}</Badge>
        <span>{item.latency}s</span>
        <span>{formatTime(item.timestamp)}</span>
      </div>
      <div className="rounded-2xl border bg-card px-4 py-3 shadow-sm">
        <div className="mb-2 text-sm font-medium text-foreground">{item.prompt}</div>
        <div className="whitespace-pre-wrap text-sm leading-6 text-foreground">{item.output}</div>
      </div>
      <div className="flex gap-2">
        <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={copy}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />} Copy
        </Button>
        <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={() => onCompare(item.prompt)}>
          <BarChart3 className="h-4 w-4" /> Compare
        </Button>
      </div>
    </div>
  );
}

function CompareCard({ row }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(row.output);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <Card className="rounded-2xl shadow-sm h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <div className="space-y-1">
            <CardTitle className="text-base capitalize">{row.mode}</CardTitle>
            <CardDescription>{row.latency}s · {row.style}</CardDescription>
          </div>
          <Badge variant="secondary" className="rounded-full px-2.5">{row.mode}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-2xl border bg-muted/20 p-4 text-sm leading-6 whitespace-pre-wrap min-h-[220px]">{row.output}</div>
        <div className="text-xs text-muted-foreground rounded-xl border p-3 space-y-1">
          <div>Temperature: {row.settings.temperature}</div>
          <div>Max tokens: {row.settings.max_tokens}</div>
        </div>
        <Button variant="outline" size="sm" className="rounded-xl gap-2" onClick={copy}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />} Copy
        </Button>
      </CardContent>
    </Card>
  );
}

export default function ModalUIApp() {
  const [style, setStyle] = useState("human");
  const [prompt, setPrompt] = useState("Help me calmly compare two job options.");
  const [forcedMode, setForcedMode] = useState("auto");
  const [batchResults, setBatchResults] = useState([]);
  const [compareResults, setCompareResults] = useState([]);
  const [history, setHistory] = useState([]);
  const [showLibrary, setShowLibrary] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversationSearch, setConversationSearch] = useState("");
  const [activeConversationId, setActiveConversationId] = useState(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(RESULTS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
          if (parsed.length > 0) setActiveConversationId(parsed[0].id);
        }
      }
    } catch {
      // ignore parse failure
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(RESULTS_KEY, JSON.stringify(history));
  }, [history]);

  const addHistoryItem = (item) => {
    setHistory((prev) => [item, ...prev].slice(0, 300));
    setActiveConversationId(item.id);
  };

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
      setError(err.message || "Request failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const runCompare = async (incomingPrompt = prompt) => {
    setIsLoading(true);
    setError("");
    try {
      const gate = chooseMode(incomingPrompt);
      const rows = [];
      for (const mode of ["home", "analytic", "engagement"]) {
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
        title: "TEST_PROMPTS batch",
        prompt: "TEST_PROMPTS batch",
        style,
        rows,
      };

      setBatchResults(rows);
      setCompareResults([]);
      addHistoryItem(batchGroup);
    } catch (err) {
      setError(err.message || "Batch run failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const conversations = useMemo(() => {
    const q = conversationSearch.trim().toLowerCase();
    return history.filter((item) => {
      if (!q) return true;
      return (item.title || item.prompt || "").toLowerCase().includes(q);
    });
  }, [history, conversationSearch]);

  const activeConversation = useMemo(() => {
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
      <div className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="mx-auto max-w-[1600px] px-4 md:px-6 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <Button variant="outline" size="icon" className="rounded-xl" onClick={() => setSidebarOpen((v) => !v)}>
              <PanelLeft className="h-4 w-4" />
            </Button>
            <div>
              <div className="text-lg font-semibold tracking-tight">Modal Daily Driver</div>
              <div className="text-xs text-muted-foreground">Shared base model · gated modes · live local backend</div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill icon={Brain} label="Model" value={MODEL} />
            <StatusPill icon={style === "human" ? HeartHandshake : Snowflake} label="Style" value={style} />
            <StatusPill icon={Terminal} label="Backend" value="Connected local" />
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-[1600px] p-4 md:p-6">
        <div className="grid grid-cols-1 xl:grid-cols-[280px_minmax(0,1fr)] gap-6">
          {sidebarOpen && (
            <Card className="rounded-3xl shadow-sm xl:h-[calc(100vh-130px)] xl:sticky xl:top-[92px] overflow-hidden">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <CardTitle className="text-base">Conversations</CardTitle>
                    <CardDescription>Single runs, comparisons, and batches</CardDescription>
                  </div>
                  <Button variant="outline" size="sm" className="rounded-xl" onClick={() => setActiveConversationId(null)}>
                    <Plus className="h-4 w-4 mr-1" /> New
                  </Button>
                </div>
                <div className="relative mt-2">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    value={conversationSearch}
                    onChange={(e) => setConversationSearch(e.target.value)}
                    placeholder="Search conversations"
                    className="pl-9 rounded-2xl"
                  />
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <ScrollArea className="h-[520px] xl:h-[calc(100vh-280px)]">
                  <div className="px-3 pb-3 space-y-2">
                    {conversations.length > 0 ? conversations.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => setActiveConversationId(item.id)}
                        className={`w-full text-left rounded-2xl border px-4 py-3 transition hover:bg-muted/40 ${activeConversationId === item.id ? "bg-muted border-foreground/10" : "bg-background"}`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="truncate text-sm font-medium">{item.title || item.prompt}</div>
                            <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock3 className="h-3 w-3" />
                              <span>{formatTime(item.timestamp)}</span>
                            </div>
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                        </div>
                        <div className="mt-2 flex gap-2 flex-wrap">
                          <Badge variant="outline" className="rounded-full px-2">{item.type}</Badge>
                          {item.style && <Badge variant="outline" className="rounded-full px-2">{item.style}</Badge>}
                        </div>
                      </button>
                    )) : (
                      <div className="text-sm text-muted-foreground rounded-2xl border p-6 text-center">No saved conversations yet.</div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          <div className="space-y-6">
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
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between gap-4 flex-wrap">
                      <div>
                        <CardTitle className="text-xl">Chat</CardTitle>
                        <CardDescription>Main daily driver window with live backend responses.</CardDescription>
                      </div>
                      <div className="flex items-center gap-3 flex-wrap">
                        <Select value={style} onValueChange={setStyle}>
                          <SelectTrigger className="w-[130px] rounded-xl"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="human">Human</SelectItem>
                            <SelectItem value="cold">Cold</SelectItem>
                          </SelectContent>
                        </Select>
                        <Select value={forcedMode} onValueChange={setForcedMode}>
                          <SelectTrigger className="w-[170px] rounded-xl"><SelectValue /></SelectTrigger>
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
                  <CardContent className="space-y-5">
                    {showLibrary && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between gap-4">
                          <Label className="text-sm font-medium">Prompt Library</Label>
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <span>Show library</span>
                            <Switch checked={showLibrary} onCheckedChange={setShowLibrary} />
                          </div>
                        </div>
                        <div className="grid md:grid-cols-2 2xl:grid-cols-3 gap-3">
                          {Object.entries(PROMPT_LIBRARY).map(([name, value]) => (
                            <button
                              key={name}
                              onClick={() => setPrompt(value)}
                              className="text-left rounded-2xl border px-4 py-3 hover:bg-muted/30 transition"
                            >
                              <div className="text-sm font-medium">{name}</div>
                              <div className="text-xs text-muted-foreground mt-1 leading-5">{value}</div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    <Separator />

                    <div className="rounded-3xl border bg-muted/10 p-4 md:p-5 space-y-4">
                      <ScrollArea className="h-[320px] pr-4">
                        <div className="space-y-5">
                          {activeConversation ? activeMessages.map((item) => (
                            <MessageBubble key={item.id} item={item} onCompare={runCompare} />
                          )) : (
                            <div className="rounded-2xl border border-dashed p-10 text-center text-muted-foreground">
                              <MessageSquare className="mx-auto h-8 w-8 mb-3 opacity-60" />
                              Open a saved conversation on the left, or run a new prompt below.
                            </div>
                          )}
                        </div>
                      </ScrollArea>

                      <div className="space-y-3">
                        <Label>Compose</Label>
                        <Textarea
                          value={prompt}
                          onChange={(e) => setPrompt(e.target.value)}
                          className="min-h-[120px] rounded-2xl"
                          placeholder="Type a prompt or choose one from the library..."
                        />
                        <div className="flex flex-wrap gap-3">
                          <Button onClick={runSingle} disabled={isLoading} className="rounded-2xl gap-2">
                            <Play className="h-4 w-4" /> {isLoading ? "Running..." : "Run Prompt"}
                          </Button>
                          <Button variant="outline" onClick={() => runCompare(prompt)} disabled={isLoading} className="rounded-2xl gap-2">
                            <BarChart3 className="h-4 w-4" /> Compare Modes
                          </Button>
                          <Button variant="outline" onClick={runBatch} disabled={isLoading} className="rounded-2xl gap-2">
                            <Library className="h-4 w-4" /> Run Batch
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="compare" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Compare</CardTitle>
                    <CardDescription>Side-by-side Home, Analytic, and Engagement outputs for the same prompt.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex gap-3">
                      <Button onClick={() => runCompare(prompt)} disabled={isLoading} className="rounded-2xl gap-2">
                        <Sparkles className="h-4 w-4" /> {isLoading ? "Comparing..." : "Run Comparison"}
                      </Button>
                    </div>
                    <div className="grid lg:grid-cols-3 gap-4">
                      {compareResults.length > 0 ? compareResults.map((row) => (
                        <CompareCard key={row.id} row={row} />
                      )) : (
                        <div className="text-sm text-muted-foreground col-span-full rounded-2xl border p-10 text-center">Run a comparison to see side-by-side outputs.</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="batch" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle>Batch Runner</CardTitle>
                    <CardDescription>Run the default test suite and inspect routing at a glance.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <Button onClick={runBatch} disabled={isLoading} className="rounded-2xl gap-2">
                        <Play className="h-4 w-4" /> {isLoading ? "Running Batch..." : "Run Batch"}
                      </Button>
                      <div className="flex gap-2 flex-wrap">
                        <Badge variant="outline" className="rounded-full px-3 py-1">Home {modeSummary.home}</Badge>
                        <Badge variant="outline" className="rounded-full px-3 py-1">Analytic {modeSummary.analytic}</Badge>
                        <Badge variant="outline" className="rounded-full px-3 py-1">Engagement {modeSummary.engagement}</Badge>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {batchResults.length > 0 ? batchResults.map((row) => (
                        <div key={row.id} className="rounded-2xl border p-4 space-y-2 bg-card">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <div className="text-sm font-medium">{row.prompt}</div>
                              <div className="text-xs text-muted-foreground mt-1">{row.mode} · {row.latency}s</div>
                            </div>
                            <Badge variant="secondary" className="rounded-full px-2.5">{row.mode}</Badge>
                          </div>
                          <div className="text-sm text-muted-foreground whitespace-pre-wrap leading-6">{row.output}</div>
                        </div>
                      )) : (
                        <div className="text-sm text-muted-foreground rounded-2xl border p-10 text-center">Run the batch to populate this view.</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="history" className="space-y-6">
                <Card className="rounded-3xl shadow-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><History className="h-5 w-5" /> History</CardTitle>
                    <CardDescription>Persistent local history of prompts, comparisons, and batch sessions.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-[620px] pr-4">
                      <div className="space-y-4">
                        {history.length > 0 ? history.map((item) => (
                          <Card key={item.id} className="rounded-2xl shadow-sm">
                            <CardContent className="p-4 space-y-3">
                              <div className="flex items-start justify-between gap-4">
                                <div>
                                  <div className="text-sm font-medium">{item.title || item.prompt}</div>
                                  <div className="text-xs text-muted-foreground mt-1">{formatTime(item.timestamp)} · {item.type} · {item.style}</div>
                                </div>
                                <Badge variant="outline" className="rounded-full px-2.5">{item.type}</Badge>
                              </div>

                              {item.type === "single" && (
                                <div className="space-y-2">
                                  <Badge variant="secondary" className="rounded-full px-2.5">{item.mode}</Badge>
                                  <div className="text-sm whitespace-pre-wrap text-muted-foreground leading-6">{item.output}</div>
                                </div>
                              )}

                              {item.type === "comparison" && (
                                <div className="grid lg:grid-cols-3 gap-3">
                                  {item.rows.map((row) => (
                                    <div key={row.id || row.mode} className="rounded-xl border p-3 space-y-2 bg-muted/10">
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium capitalize">{row.mode}</span>
                                        <Badge variant="secondary" className="rounded-full px-2.5">{row.mode}</Badge>
                                      </div>
                                      <div className="text-xs text-muted-foreground whitespace-pre-wrap leading-5">{row.output}</div>
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
                          <div className="text-sm text-muted-foreground rounded-2xl border p-10 text-center">No history yet. Run a prompt, comparison, or batch to populate this tab.</div>
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
                    <CardDescription>Live configuration and exact prompts sent to the backend.</CardDescription>
                  </CardHeader>
                  <CardContent className="grid lg:grid-cols-2 gap-6">
                    <div className="rounded-2xl border p-4 space-y-3 text-sm bg-muted/10">
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Model</span><span className="font-medium text-right">{MODEL}</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Base URL</span><span className="font-medium text-right">{BASE_URL}</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Results</span><span className="font-medium text-right">localStorage</span></div>
                      <div className="flex justify-between gap-4"><span className="text-muted-foreground">Style</span><span className="font-medium text-right capitalize">{style}</span></div>
                    </div>
                    <div className="rounded-2xl border p-4 space-y-3 text-sm bg-muted/10">
                      <div className="font-medium">Mode Prompts</div>
                      <div className="space-y-3 leading-6 text-muted-foreground">
                        <div><span className="font-medium text-foreground">Home:</span> {modePrompts[style].home}</div>
                        <div><span className="font-medium text-foreground">Analytic:</span> {modePrompts[style].analytic}</div>
                        <div><span className="font-medium text-foreground">Engagement:</span> {modePrompts[style].engagement}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}

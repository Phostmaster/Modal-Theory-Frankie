import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Brain, Terminal, Layers3, Sparkles, HeartHandshake, Snowflake, Play, Library, BarChart3, Settings2, Copy, Check, FileText } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";

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
  grounded_thinking: "Can you help me think through this in a grounded way.",
};

const ANALYTIC_KEYWORDS = [
  "compare","difference","differences","why","how","evidence","assumption","assumptions","analyze","analysis","test","debug","plan","explain","causation","correlation","model","mechanism","evaluate","reason","reasoning","distinguish","strengths","weaknesses","pros","cons",
];

const ENGAGEMENT_KEYWORDS = [
  "help me","i feel","i'm feeling","im feeling","overwhelmed","worried","stay with me","gently","support","supportive","calm","talk through","with me","kind","human","present","honest answer","grounded","gentle","steadiness","care","companionship","upset",
];

const HOME_KEYWORDS = [
  "what is","who is","when is","where is","capital","define","name","list","give me","tell me","who wrote",
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
  return { temperature: 0.7, max_tokens: 220 };
}

function mockResponse(prompt, mode, style) {
  const openers = {
    home: style === "cold" ? "Please provide the relevant details." : "Of course. ",
    analytic: style === "cold" ? "Please provide the relevant inputs for analysis." : "Of course. ",
    engagement: style === "cold" ? "Please describe the situation clearly." : "I'm here with you. ",
  };

  if (/capital of france/i.test(prompt)) return "The capital of France is Paris.";
  if (/compare two job options/i.test(prompt)) {
    if (mode === "home") return `${openers[mode]}Share salary, location, responsibilities, benefits, work-life balance, and growth potential for both jobs, and I will compare them clearly.`;
    if (mode === "analytic") return `${openers[mode]}Provide both roles, compensation, work environment, growth opportunities, and your priorities. I will compare them step by step across objective criteria.`;
    return `${openers[mode]}Tell me about both jobs and what matters most to you right now, and we will work through them calmly side by side.`;
  }
  if (/upset/i.test(prompt) && /plan/i.test(prompt)) {
    if (mode === "home") return "Name the issue, identify what is true right now, choose one small action in the next 24 hours, and then reassess.";
    if (mode === "analytic") return "Define the problem, separate facts from interpretations, list constraints, generate 2–3 actions, and choose the lowest-cost next step.";
    return "You do not need to solve everything at once. Tell me what happened, what hurts most, and what needs doing first, and we will make a grounded plan together.";
  }
  if (mode === "home") return `${openers[mode]}I can help with that directly. Give me the key detail and I will keep it clear and brief.`;
  if (mode === "analytic") return `${openers[mode]}I can break that down carefully. Share the details and I will structure the reasoning clearly.`;
  return `${openers[mode]}Tell me a little more about what you need, and I will stay clear, steady, and practical with you.`;
}

function Stat({ label, value, icon: Icon }) {
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardContent className="p-4 flex items-center gap-3">
        <div className="p-2 rounded-2xl bg-muted"><Icon className="w-4 h-4" /></div>
        <div>
          <div className="text-xs text-muted-foreground">{label}</div>
          <div className="text-sm font-semibold">{value}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function OutputCard({ title, mode, style, output, scores, settings, prompt, compact = false }) {
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
  const [model] = useState("qwen3-4b-instruct-2507.gguf");
  const [baseUrl] = useState("http://127.0.0.1:1234/v1/chat/completions");
  const [resultsDir] = useState("results");
  const [prompt, setPrompt] = useState("Help me calmly compare two job options.");
  const [forcedMode, setForcedMode] = useState("auto");
  const [lastResult, setLastResult] = useState(null);
  const [compareResults, setCompareResults] = useState([]);
  const [batchResults, setBatchResults] = useState([]);
  const [showLibrary, setShowLibrary] = useState(true);

  const status = useMemo(() => ({ model, baseUrl, resultsDir, style }), [model, baseUrl, resultsDir, style]);

  const runSingle = () => {
    const gate = chooseMode(prompt);
    const mode = forcedMode === "auto" ? gate.mode : forcedMode;
    const settings = getGenerationSettings(mode);
    const output = mockResponse(prompt, mode, style);
    setLastResult({ prompt, mode, scores: gate.scores, settings, output, style });
    setCompareResults([]);
  };

  const runCompare = () => {
    const gate = chooseMode(prompt);
    const rows = ["home", "analytic", "engagement"].map((mode) => ({
      prompt,
      mode,
      scores: gate.scores,
      settings: getGenerationSettings(mode),
      output: mockResponse(prompt, mode, style),
      style,
    }));
    setCompareResults(rows);
    setLastResult(null);
  };

  const runBatch = () => {
    const rows = TEST_PROMPTS.map((p) => {
      const gate = chooseMode(p);
      const mode = gate.mode;
      return {
        prompt: p,
        mode,
        scores: gate.scores,
        output: mockResponse(p, mode, style),
        settings: getGenerationSettings(mode),
      };
    });
    setBatchResults(rows);
    setLastResult(null);
    setCompareResults([]);
  };

  const loadLibraryPrompt = (value) => setPrompt(value);

  const modeSummary = useMemo(() => {
    const src = batchResults.length ? batchResults : TEST_PROMPTS.map((p) => ({ mode: chooseMode(p).mode }));
    return src.reduce((acc, row) => {
      acc[row.mode] = (acc[row.mode] || 0) + 1;
      return acc;
    }, { home: 0, analytic: 0, engagement: 0 });
  }, [batchResults]);

  return (
    <div className="min-h-screen bg-background text-foreground p-6 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="grid lg:grid-cols-[1.2fr_0.8fr] gap-6">
          <Card className="rounded-3xl shadow-sm overflow-hidden">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-3 rounded-2xl bg-muted"><Layers3 className="w-5 h-5" /></div>
                <div>
                  <CardTitle className="text-2xl">Modal Orchestrator UI</CardTitle>
                  <CardDescription>Shared base model, lightweight gate, three privilege modes, optional human or cold style.</CardDescription>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary" className="rounded-xl">Home / Base</Badge>
                <Badge variant="secondary" className="rounded-xl">Analytic / Reasoning</Badge>
                <Badge variant="secondary" className="rounded-xl">Relational / Engagement</Badge>
              </div>
            </CardHeader>
            <CardContent className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
              <Stat label="Model" value={model} icon={Brain} />
              <Stat label="Base URL" value="Local endpoint" icon={Terminal} />
              <Stat label="Style" value={style} icon={style === "human" ? HeartHandshake : Snowflake} />
              <Stat label="Results Dir" value={resultsDir} icon={FileText} />
            </CardContent>
          </Card>

          <Card className="rounded-3xl shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Settings2 className="w-5 h-5" /> Session Controls</CardTitle>
              <CardDescription>Choose style, mode behavior, and whether to work from the prompt library.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
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
              </div>
              <div className="flex items-center justify-between rounded-2xl border p-4">
                <div>
                  <div className="font-medium">Prompt Library</div>
                  <div className="text-sm text-muted-foreground">Show ready-made test prompts for fast runs.</div>
                </div>
                <Switch checked={showLibrary} onCheckedChange={setShowLibrary} />
              </div>
              <div className="text-xs text-muted-foreground rounded-2xl bg-muted/40 p-3">
                Style in use: <span className="font-medium text-foreground">{style}</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <Tabs defaultValue="workbench" className="space-y-6">
          <TabsList className="grid grid-cols-4 w-full max-w-2xl rounded-2xl">
            <TabsTrigger value="workbench">Workbench</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
            <TabsTrigger value="batch">Batch</TabsTrigger>
            <TabsTrigger value="status">Status</TabsTrigger>
          </TabsList>

          <TabsContent value="workbench" className="space-y-6">
            <Card className="rounded-3xl shadow-sm">
              <CardHeader>
                <CardTitle>Single Prompt</CardTitle>
                <CardDescription>Run one prompt through the gate or a forced mode.</CardDescription>
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
                  <Button onClick={runSingle} className="rounded-2xl gap-2"><Play className="w-4 h-4" /> Run Prompt</Button>
                  <Button variant="outline" onClick={runCompare} className="rounded-2xl gap-2"><BarChart3 className="w-4 h-4" /> Compare Modes</Button>
                  <Button variant="outline" onClick={runBatch} className="rounded-2xl gap-2"><Library className="w-4 h-4" /> Run Batch</Button>
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
              />
            )}
          </TabsContent>

          <TabsContent value="compare" className="space-y-6">
            <Card className="rounded-3xl shadow-sm">
              <CardHeader>
                <CardTitle>Forced Mode Comparison</CardTitle>
                <CardDescription>See the same prompt through Home, Analytic, and Engagement modes.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-3 mb-4">
                  <Button onClick={runCompare} className="rounded-2xl gap-2"><Sparkles className="w-4 h-4" /> Run Comparison</Button>
                </div>
                <div className="grid lg:grid-cols-3 gap-4">
                  {compareResults.length > 0 ? compareResults.map((row) => (
                    <OutputCard
                      key={row.mode}
                      title={row.mode.charAt(0).toUpperCase() + row.mode.slice(1)}
                      mode={row.mode}
                      style={row.style}
                      output={row.output}
                      scores={row.scores}
                      settings={row.settings}
                      prompt={row.prompt}
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
                <CardDescription>Run the default prompt suite and inspect routing decisions.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-3">
                  <Button onClick={runBatch} className="rounded-2xl gap-2"><Play className="w-4 h-4" /> Run Batch</Button>
                </div>
                <div className="grid md:grid-cols-3 gap-4">
                  <Stat label="Home" value={modeSummary.home} icon={Layers3} />
                  <Stat label="Analytic" value={modeSummary.analytic} icon={Brain} />
                  <Stat label="Engagement" value={modeSummary.engagement} icon={HeartHandshake} />
                </div>
                <Separator />
                <div className="space-y-3">
                  {batchResults.length > 0 ? batchResults.map((row, idx) => (
                    <Card key={`${row.prompt}-${idx}`} className="rounded-2xl">
                      <CardContent className="p-4 space-y-3">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <div className="text-sm font-medium">{row.prompt}</div>
                            <div className="text-xs text-muted-foreground mt-1">Mode selected: {row.mode}</div>
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

          <TabsContent value="status" className="space-y-6">
            <Card className="rounded-3xl shadow-sm">
              <CardHeader>
                <CardTitle>Current Status</CardTitle>
                <CardDescription>Session configuration and runtime posture.</CardDescription>
              </CardHeader>
              <CardContent className="grid lg:grid-cols-2 gap-6">
                <div className="rounded-2xl border p-4 space-y-3 text-sm">
                  <div className="flex justify-between gap-4"><span className="text-muted-foreground">Model</span><span className="font-medium text-right">{status.model}</span></div>
                  <div className="flex justify-between gap-4"><span className="text-muted-foreground">Base URL</span><span className="font-medium text-right">{status.baseUrl}</span></div>
                  <div className="flex justify-between gap-4"><span className="text-muted-foreground">Results Dir</span><span className="font-medium text-right">{status.resultsDir}</span></div>
                  <div className="flex justify-between gap-4"><span className="text-muted-foreground">Style</span><span className="font-medium text-right capitalize">{status.style}</span></div>
                </div>
                <div className="rounded-2xl border p-4 space-y-3 text-sm">
                  <div className="font-medium">Mode Prompts</div>
                  <div className="space-y-2">
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
    </div>
  );
}

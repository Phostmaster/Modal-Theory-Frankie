npm install -g expo-cli               # only once globally
npx create-expo-app orchestrator-native
cd orchestrator-native

npm install react-native-paper react-native-gesture-handler react-native-reanimated react-native-safe-area-context react-native-vector-icons

npx expo install react-native-gesture-handler react-native-reanimated react-native-safe-area-context

import React, { useState } from "react";
import { SafeAreaView, ScrollView, StyleSheet, View } from "react-native";
import {
  Provider as PaperProvider,
  Card,
  Text,
  Button,
  Chip,
  TextInput,
  SegmentedButtons,
  ActivityIndicator,
} from "react-native-paper";
import { chooseMode, modePrompts, getGenerationSettings, callBackend } from "./src/orchestration"; // move logic into RN-friendly modules

const PROMPT_LIBRARY = [
  "Help me calmly compare two job options.",
  "What is the capital of France?",
  "Explain this clearly without overcomplicating it.",
  // ...same list
];

export default function App() {
  const [style, setStyle] = useState("human");
  const [prompt, setPrompt] = useState(PROMPT_LIBRARY[0]);
  const [forcedMode, setForcedMode] = useState("auto");
  const [isLoading, setIsLoading] = useState(false);
  const [output, setOutput] = useState("");
  const [history, setHistory] = useState([]);

  const runSingle = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    try {
      const gate = chooseMode(prompt);
      const mode = forcedMode === "auto" ? gate.mode : forcedMode;
      const systemPrompt = modePrompts[style][mode];
      const result = await callBackend({ systemPrompt, userPrompt: prompt, mode });

      setOutput(result.output);

      setHistory((prev) => [
        { id: Date.now().toString(), prompt, output: result.output, mode },
        ...prev,
      ]);
    } catch (err) {
      setOutput(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <PaperProvider>
      <SafeAreaView style={styles.safe}>
        <ScrollView contentContainerStyle={styles.container}>
          <Card style={styles.card}>
            <Card.Title title="Orchestrator Studio" subtitle="Expo React Native build" />
            <Card.Content>
              <SegmentedButtons
                style={{ marginBottom: 12 }}
                value={style}
                onValueChange={setStyle}
                buttons={[
                  { value: "human", label: "Human" },
                  { value: "cold", label: "Cold" },
                ]}
              />
              <SegmentedButtons
                value={forcedMode}
                onValueChange={setForcedMode}
                buttons={[
                  { value: "auto", label: "Auto Gate" },
                  { value: "home", label: "Home" },
                  { value: "analytic", label: "Analytic" },
                  { value: "engagement", label: "Engagement" },
                ]}
              />
              <Text style={styles.sectionLabel}>Prompt Library</Text>
              <View style={styles.libraryWrap}>
                {PROMPT_LIBRARY.map((item) => (
                  <Chip
                    key={item}
                    style={styles.chip}
                    mode={prompt === item ? "flat" : "outlined"}
                    selected={prompt === item}
                    onPress={() => setPrompt(item)}
                  >
                    {item.slice(0, 40)}
                  </Chip>
                ))}
              </View>
              <TextInput
                label="Prompt"
                mode="outlined"
                multiline
                value={prompt}
                onChangeText={setPrompt}
                style={{ marginVertical: 16 }}
              />
              <Button mode="contained" onPress={runSingle} disabled={isLoading}>
                {isLoading ? <ActivityIndicator animating /> : "Run Prompt"}
              </Button>
              {output ? (
                <>
                  <Text style={styles.sectionLabel}>Output</Text>
                  <Card mode="outlined">
                    <Card.Content>
                      <Text>{output}</Text>
                    </Card.Content>
                  </Card>
                </>
              ) : null}
            </Card.Content>
          </Card>

          {history.length > 0 && (
            <Card style={styles.card}>
              <Card.Title title="History" />
              <Card.Content>
                {history.map((entry) => (
                  <View key={entry.id} style={styles.historyItem}>
                    <Text style={styles.historyPrompt}>{entry.prompt}</Text>
                    <Text style={styles.historyMode}>Mode: {entry.mode}</Text>
                    <Text>{entry.output}</Text>
                  </View>
                ))}
              </Card.Content>
            </Card>
          )}
        </ScrollView>
      </SafeAreaView>
    </PaperProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#0B0F19" },
  container: { padding: 16, gap: 16 },
  card: { borderRadius: 20 },
  sectionLabel: { marginTop: 16, marginBottom: 8, fontWeight: "600" },
  libraryWrap: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: { borderRadius: 16 },
  historyItem: { marginBottom: 16, paddingBottom: 16, borderBottomWidth: 1, borderColor: "rgba(255,255,255,0.05)" },
  historyPrompt: { fontWeight: "600" },
  historyMode: { color: "rgba(255,255,255,0.6)", marginBottom: 6 },
});

npx expo start

npx expo prebuild         # sets up the native Android project
npx expo run:android --variant release   # builds and installs a release build locally

npx expo install eas-cli
npx eas build -p android
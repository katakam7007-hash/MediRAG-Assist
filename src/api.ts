import { GeminiModelInfo } from './types';

export interface HealthResponse {
  status: string;
  hasApiKey: boolean;
  timestamp: string;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchGeminiModels(): Promise<{
  success: boolean;
  models: GeminiModelInfo[];
  defaultGenerationModel: string;
  defaultEmbeddingModel: string;
}> {
  const res = await fetch('/api/gemini/models');
  if (!res.ok) throw new Error(`Fetch models failed: ${res.statusText}`);
  return res.json();
}

export async function embedTexts(
  texts: string[],
  docName?: string,
  query?: boolean,
  model?: string
): Promise<{ success: boolean; embeddings: number[][]; dimension: number }> {
  const res = await fetch('/api/gemini/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, docName, query, model }),
  });
  const data = await res.json();
  if (!res.ok || !data.success) {
    throw new Error(data.error || 'Embedding generation failed');
  }
  return data;
}

export async function rerankPassages(
  query: string,
  passages: string[],
  model?: string
): Promise<{ success: boolean; scores: number[] }> {
  const res = await fetch('/api/gemini/rerank', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, passages, model }),
  });
  return res.json();
}

export async function generateContent(payload: {
  model?: string;
  contents: any;
  systemInstruction?: string;
  temperature?: number;
  image?: { base64: string; mimeType: string };
}): Promise<{
  success: boolean;
  text: string;
  modelUsed: string;
  fallbackOccurred: boolean;
  latency: number;
  error?: string;
}> {
  const res = await fetch('/api/gemini/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok && !data.text) {
    throw new Error(data.error || `Generation error (${res.status})`);
  }
  return data;
}

export async function streamContent(
  payload: {
    model?: string;
    contents: any;
    systemInstruction?: string;
    temperature?: number;
    image?: { base64: string; mimeType: string };
  },
  callbacks: {
    onStart?: (modelUsed: string) => void;
    onChunk: (text: string) => void;
    onFallback?: (notice: string) => void;
    onError: (err: string) => void;
    onDone: (modelUsed: string) => void;
  }
): Promise<void> {
  const res = await fetch('/api/gemini/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Stream request failed (${res.status})`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No readable stream available');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data: ')) continue;
      const dataStr = trimmed.replace(/^data:\s*/, '');
      if (dataStr === '[DONE]') {
        callbacks.onDone('');
        return;
      }
      try {
        const parsed = JSON.parse(dataStr);
        if (parsed.type === 'start') {
          callbacks.onStart?.(parsed.modelUsed);
        } else if (parsed.type === 'chunk') {
          callbacks.onChunk(parsed.text);
        } else if (parsed.type === 'fallback_notice') {
          callbacks.onFallback?.(parsed.message);
        } else if (parsed.type === 'error') {
          callbacks.onError(parsed.error);
        } else if (parsed.type === 'done') {
          callbacks.onDone(parsed.modelUsed);
        }
      } catch {
        // Skip unparseable lines
      }
    }
  }
}

export async function simulateError(errorType: '429' | '503'): Promise<any> {
  const res = await fetch('/api/gemini/simulate-error', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ errorType }),
  });
  return res.json();
}

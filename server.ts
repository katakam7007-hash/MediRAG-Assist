import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const isProduction = process.env.NODE_ENV === 'production';

app.use(express.json({ limit: '25mb' }));

// Initialize GoogleGenAI
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = new GoogleGenAI({
  apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Primary & Fallback Models
const DEFAULT_GENERATION_MODEL = 'gemini-3.8-flash';
const FALLBACK_MODELS = [
  'gemini-2.5-flash',
  'gemini-3.5-flash',
  'gemini-3.1-flash-lite',
  'gemini-flash-latest',
];
const DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-2';

// Helper: Translate raw API errors into plain language
function mapErrorToFriendlyMessage(error: unknown): { status: number; message: string; code: string } {
  const errStr = error instanceof Error ? error.message : String(error);

  if (errStr.includes('429') || errStr.toLowerCase().includes('quota') || errStr.toLowerCase().includes('rate limit')) {
    return {
      status: 429,
      code: 'RATE_LIMIT_EXCEEDED',
      message: 'Gemini quota or rate limit reached for this model; wait a minute or choose another model in Settings.',
    };
  }
  if (errStr.includes('503') || errStr.toLowerCase().includes('overloaded') || errStr.toLowerCase().includes('unavailable') || errStr.toLowerCase().includes('high demand')) {
    return {
      status: 503,
      code: 'SERVICE_OVERLOADED',
      message: 'Gemini model is currently experiencing high demand. Automatically falling back to an alternate model.',
    };
  }
  if (errStr.includes('400') || errStr.includes('403') || errStr.toLowerCase().includes('permission') || errStr.toLowerCase().includes('key')) {
    return {
      status: 403,
      code: 'REQUEST_OR_KEY_PROBLEM',
      message: 'Gemini request or key authentication issue. Please verify API configuration in AI Studio Secrets.',
    };
  }
  if (errStr.includes('404') || errStr.toLowerCase().includes('not found')) {
    return {
      status: 404,
      code: 'MODEL_NOT_FOUND',
      message: 'The selected Gemini model is not found or unsupported in this region. Falling back to an alternate model.',
    };
  }
  if (errStr.toLowerCase().includes('fetch failed') || errStr.toLowerCase().includes('network') || errStr.toLowerCase().includes('timeout')) {
    return {
      status: 504,
      code: 'NETWORK_TIMEOUT',
      message: 'Network connection error or timeout reaching Gemini. Please check your connection.',
    };
  }

  return {
    status: 500,
    code: 'GENERIC_ERROR',
    message: 'An unexpected error occurred while communicating with Gemini. Please try again.',
  };
}

// Retry helper for transient 500/502/504 errors (NOT 429 or 503 high demand which trigger model fallback)
async function callWithRetry<T>(fn: () => Promise<T>, maxRetries = 1): Promise<T> {
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      return await fn();
    } catch (err: unknown) {
      const errStr = err instanceof Error ? err.message : String(err);
      const isTransient5xx = errStr.includes('500') || errStr.includes('502') || errStr.includes('504');

      if (!isTransient5xx || attempt === maxRetries) {
        throw err;
      }
      attempt++;
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
  throw new Error('Max retries exceeded');
}

// 1. Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hasApiKey: Boolean(apiKey && apiKey.length > 5),
    timestamp: new Date().toISOString(),
  });
});

// 2. List available models
app.get('/api/gemini/models', async (req, res) => {
  try {
    const pager = await ai.models.list();
    const models: Array<{ name: string; displayName?: string }> = [];
    for await (const m of pager) {
      const name = (m.name || '').replace(/^models\//, '');
      if (name.includes('flash') || name.includes('embed')) {
        models.push({
          name,
          displayName: m.displayName || name,
        });
      }
    }
    res.json({
      success: true,
      models,
      defaultGenerationModel: DEFAULT_GENERATION_MODEL,
      defaultEmbeddingModel: DEFAULT_EMBEDDING_MODEL,
    });
  } catch (error: unknown) {
    const friendly = mapErrorToFriendlyMessage(error);
    res.json({
      success: false,
      error: friendly.message,
      models: [
        { name: 'gemini-3.8-flash', displayName: 'Gemini 3.8 Flash' },
        { name: 'gemini-2.5-flash', displayName: 'Gemini 2.5 Flash' },
        { name: 'gemini-3.1-flash-lite', displayName: 'Gemini 3.1 Flash Lite' },
        { name: 'gemini-flash-latest', displayName: 'Gemini Flash Latest' },
      ],
      defaultGenerationModel: DEFAULT_GENERATION_MODEL,
      defaultEmbeddingModel: DEFAULT_EMBEDDING_MODEL,
    });
  }
});

// 3. Embedding Endpoint
app.post('/api/gemini/embed', async (req, res) => {
  const { model = DEFAULT_EMBEDDING_MODEL, texts, taskType, docName, query } = req.body;

  if (!texts || !Array.isArray(texts) || texts.length === 0) {
    res.status(400).json({ success: false, error: 'Texts array is required.' });
    return;
  }

  try {
    const isModel2 = model.includes('embedding-2');
    const vectors: number[][] = [];
    const targetModel = model || DEFAULT_EMBEDDING_MODEL;

    // Process each text to prevent aggregation bug
    for (let i = 0; i < texts.length; i++) {
      const rawText = texts[i];
      let formattedText = rawText;

      if (isModel2) {
        if (query) {
          formattedText = `task: question answering | query: ${rawText}`;
        } else if (docName) {
          formattedText = `title: ${docName} | text: ${rawText}`;
        }
      }

      const embedRes = await callWithRetry(async () => {
        return await ai.models.embedContent({
          model: targetModel,
          contents: {
            parts: [{ text: formattedText }],
          },
          config: {
            outputDimensionality: 768,
            ...(isModel2
              ? {}
              : { taskType: taskType || (query ? 'QUESTION_ANSWERING' : 'RETRIEVAL_DOCUMENT') }),
          },
        });
      });

      const values = embedRes.embeddings?.[0]?.values || (embedRes as any).embedding?.values;
      if (!values || !Array.isArray(values) || values.length === 0) {
        throw new Error(`Embedding generation returned empty vector for chunk ${i}`);
      }
      vectors.push(values);

      // Brief delay between batches to stay comfortably within rate limits
      if (texts.length > 1 && i < texts.length - 1) {
        await new Promise((r) => setTimeout(r, 60));
      }
    }

    if (vectors.length !== texts.length) {
      throw new Error(`Embeddings count (${vectors.length}) does not match input chunk count (${texts.length})`);
    }

    res.json({
      success: true,
      embeddings: vectors,
      modelUsed: targetModel,
      dimension: vectors[0]?.length || 768,
    });
  } catch (error: unknown) {
    const friendly = mapErrorToFriendlyMessage(error);
    res.status(friendly.status).json({
      success: false,
      code: friendly.code,
      error: friendly.message,
    });
  }
});

// 4. Generate Content (Unary) with Auto-Fallback & Multimodal Image Support
app.post('/api/gemini/generate', async (req, res) => {
  const {
    model = DEFAULT_GENERATION_MODEL,
    contents,
    systemInstruction,
    responseMimeType,
    responseSchema,
    temperature = 0.2,
    image,
  } = req.body;

  // Format contents for multimodal image input if image is provided
  let formattedContents = contents;
  if (image && image.base64) {
    const cleanBase64 = image.base64.replace(/^data:[^;]+;base64,/, '');
    const imagePart = {
      inlineData: {
        mimeType: image.mimeType || 'image/jpeg',
        data: cleanBase64,
      },
    };
    if (typeof contents === 'string') {
      formattedContents = {
        parts: [imagePart, { text: contents }],
      };
    } else if (contents && Array.isArray(contents.parts)) {
      formattedContents = {
        parts: [imagePart, ...contents.parts],
      };
    } else {
      formattedContents = {
        parts: [imagePart, { text: JSON.stringify(contents) }],
      };
    }
  }

  const candidateModels = [model, ...FALLBACK_MODELS.filter((m) => m !== model)];
  let lastError: unknown = null;
  const startTime = Date.now();

  for (const candidate of candidateModels) {
    try {
      const response = await callWithRetry(async () => {
        return await ai.models.generateContent({
          model: candidate,
          contents: formattedContents,
          config: {
            systemInstruction,
            temperature,
            ...(responseMimeType ? { responseMimeType } : {}),
            ...(responseSchema ? { responseSchema } : {}),
          },
        });
      });

      const latency = Date.now() - startTime;
      res.json({
        success: true,
        text: response.text || '',
        modelUsed: candidate,
        fallbackOccurred: candidate !== model,
        latency,
      });
      return;
    } catch (err: unknown) {
      lastError = err;
      const friendly = mapErrorToFriendlyMessage(err);
      console.warn(`[Model Generate Failed] Model ${candidate} failed: ${friendly.message}. Attempting fallback...`);
      // If error is 429, 503, 404, we try fallback model
      if (friendly.status === 429 || friendly.status === 503 || friendly.status === 404) {
        continue;
      } else {
        // Break for 400 or invalid parameters
        break;
      }
    }
  }

  const friendly = mapErrorToFriendlyMessage(lastError);
  res.status(friendly.status).json({
    success: false,
    code: friendly.code,
    error: friendly.message,
  });
});

// 5. Stream Content (SSE) with Auto-Fallback & Multimodal Image Support
app.post('/api/gemini/stream', async (req, res) => {
  const {
    model = DEFAULT_GENERATION_MODEL,
    contents,
    systemInstruction,
    temperature = 0.2,
    image,
  } = req.body;

  // Format contents for multimodal image input if image is provided
  let formattedContents = contents;
  if (image && image.base64) {
    const cleanBase64 = image.base64.replace(/^data:[^;]+;base64,/, '');
    const imagePart = {
      inlineData: {
        mimeType: image.mimeType || 'image/jpeg',
        data: cleanBase64,
      },
    };
    if (typeof contents === 'string') {
      formattedContents = {
        parts: [imagePart, { text: contents }],
      };
    } else if (contents && Array.isArray(contents.parts)) {
      formattedContents = {
        parts: [imagePart, ...contents.parts],
      };
    } else {
      formattedContents = {
        parts: [imagePart, { text: JSON.stringify(contents) }],
      };
    }
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const candidateModels = [model, ...FALLBACK_MODELS.filter((m) => m !== model)];
  let streamSucceeded = false;

  for (const candidate of candidateModels) {
    try {
      res.write(`data: ${JSON.stringify({ type: 'start', modelUsed: candidate, fallback: candidate !== model })}\n\n`);

      const stream = await ai.models.generateContentStream({
        model: candidate,
        contents: formattedContents,
        config: {
          systemInstruction,
          temperature,
        },
      });

      for await (const chunk of stream) {
        const text = chunk.text;
        if (text) {
          res.write(`data: ${JSON.stringify({ type: 'chunk', text })}\n\n`);
        }
      }

      res.write(`data: ${JSON.stringify({ type: 'done', modelUsed: candidate })}\n\n`);
      streamSucceeded = true;
      break;
    } catch (err: unknown) {
      const friendly = mapErrorToFriendlyMessage(err);
      console.warn(`[Stream Error with ${candidate}]:`, friendly.message);
      if (friendly.status === 429 || friendly.status === 503 || friendly.status === 404) {
        res.write(`data: ${JSON.stringify({ type: 'fallback_notice', message: `Model ${candidate} unavailable (${friendly.message}). Switching model...` })}\n\n`);
        continue;
      } else {
        res.write(`data: ${JSON.stringify({ type: 'error', error: friendly.message, code: friendly.code })}\n\n`);
        break;
      }
    }
  }

  if (!streamSucceeded) {
    res.write(`data: ${JSON.stringify({ type: 'error', error: 'All candidate models failed to stream response.', code: 'ALL_MODELS_FAILED' })}\n\n`);
  }

  res.write('data: [DONE]\n\n');
  res.end();
});

// 6. Gemini Re-ranking Endpoint (Scores passages 0-10)
app.post('/api/gemini/rerank', async (req, res) => {
  const { query, passages, model = 'gemini-3.1-flash-lite' } = req.body;
  if (!query || !Array.isArray(passages) || passages.length === 0) {
    res.json({ success: true, scores: passages.map(() => 5) });
    return;
  }

  try {
    const prompt = `You are a medical passage relevance scorer.
Given the patient query and candidate medical passages, score each passage on its clinical relevance to answering the query on a scale of 0 to 10 (10 = directly answers the query with verified facts, 0 = completely irrelevant).

Query: "${query}"

Passages:
${passages.map((p: string, idx: number) => `[Passage ${idx}]:\n${p}\n`).join('\n')}

Return JSON array of numbers representing scores for Passage 0 to Passage ${passages.length - 1}.`;

    const response = await ai.models.generateContent({
      model: model || 'gemini-3.1-flash-lite',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.NUMBER,
          },
        },
      },
    });

    const parsed = JSON.parse(response.text?.trim() || '[]');
    res.json({
      success: true,
      scores: Array.isArray(parsed) && parsed.length === passages.length ? parsed : passages.map(() => 5),
    });
  } catch (err: unknown) {
    console.warn('[Rerank fallback to neutral scores]', err);
    res.json({ success: true, scores: passages.map(() => 5) });
  }
});

// 7. Simulated Error Endpoint (For Test Lab verification)
app.post('/api/gemini/simulate-error', (req, res) => {
  const { errorType } = req.body;
  if (errorType === '429') {
    res.status(429).json({
      success: false,
      code: 'RATE_LIMIT_EXCEEDED',
      error: 'Gemini quota or rate limit reached for this model; wait a minute or choose another model in Settings.',
    });
    return;
  }
  if (errorType === '503') {
    res.status(503).json({
      success: false,
      code: 'SERVICE_OVERLOADED',
      error: 'Gemini is temporarily overloaded. Please try again shortly or switch to a lighter model.',
    });
    return;
  }
  res.status(400).json({ success: false, error: 'Invalid error simulation type.' });
});

// Start Server with Vite Dev in local development
if (process.env.NODE_ENV !== 'production') {
  const startServer = async () => {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);

    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Medi-RAG Server running on port ${PORT}`);
    });
  };

  startServer().catch((err) => {
    console.error('Failed to start server:', err);
  });
}

export default app;


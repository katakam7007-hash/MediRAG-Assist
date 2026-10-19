import React, { useState, useEffect, useRef } from 'react';
import {
  Stethoscope,
  BookOpen,
  Activity,
  Settings,
  Send,
  Upload,
  Trash2,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  RefreshCw,
  FileText,
  Brain,
  ShieldAlert,
  Image as ImageIcon,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';
import {
  checkHealth,
  fetchGeminiModels,
  embedTexts,
  rerankPassages,
  generateContent,
  streamContent,
  simulateError,
  HealthResponse,
} from './api';
import {
  MedicalDoc,
  MedicalChunk,
  RetrievedChunk,
  ChatMessage,
  GeminiModelInfo,
} from './types';
import {
  getDocuments,
  saveDocument,
  deleteDocument,
  searchChunksByVector,
  SAMPLE_GUIDELINES,
} from './db';

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'knowledge' | 'triage' | 'settings'>('chat');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<GeminiModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('gemini-3.8-flash');
  
  // Knowledge Base State
  const [documents, setDocuments] = useState<MedicalDoc[]>([]);
  const [isEmbeddingSample, setIsEmbeddingSample] = useState(false);
  const [customDocTitle, setCustomDocTitle] = useState('');
  const [customDocContent, setCustomDocContent] = useState('');
  const [isAddingDoc, setIsAddingDoc] = useState(false);

  // Chat / RAG State
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Welcome to **Medi-RAG**, your educational medical decision-support assistant.\n\nAsk any clinical question or review patient scenarios. Medi-RAG retrieves authoritative clinical guidelines from your local IndexedDB knowledge base, verifies relevance via Gemini Re-ranking, and provides syntheses with transparent citations.',
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [queryInput, setQueryInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [imageAttachment, setImageAttachment] = useState<{ base64: string; mimeType: string; preview: string } | null>(null);
  const [currentRetrievedPassages, setCurrentRetrievedPassages] = useState<RetrievedChunk[]>([]);
  const [useStreaming, setUseStreaming] = useState(true);

  // Triage / Differential Lab State
  const [triageAge, setTriageAge] = useState('54');
  const [triageSex, setTriageSex] = useState('Male');
  const [triageComplaint, setTriageComplaint] = useState('Sudden onset shortness of breath and wheezing after outdoor exertion');
  const [triageVitals, setTriageVitals] = useState('BP 142/88, HR 105, RR 24, SpO2 91% on room air, Temp 37.1 C');
  const [triageRedFlags, setTriageRedFlags] = useState('Accessory muscle use, unable to speak in full sentences');
  const [triageResult, setTriageResult] = useState('');
  const [isTriaging, setIsTriaging] = useState(false);

  // Test Lab State
  const [testResult, setTestResult] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initial load
  useEffect(() => {
    refreshHealth();
    loadModels();
    loadDocs();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  async function refreshHealth() {
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch (err: any) {
      setHealth({ status: 'error', hasApiKey: false, timestamp: new Date().toISOString() });
    }
  }

  async function loadModels() {
    try {
      const data = await fetchGeminiModels();
      if (data.models && data.models.length > 0) {
        setModels(data.models);
        setSelectedModel(data.defaultGenerationModel || data.models[0].name);
      }
    } catch (err) {
      console.warn('Could not load dynamic models, using fallback.');
    }
  }

  async function loadDocs() {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
      // Auto-load sample guidelines if empty
      if (docs.length === 0) {
        await initSampleKnowledge();
      }
    } catch (err) {
      console.error('Error loading documents:', err);
    }
  }

  async function initSampleKnowledge() {
    setIsEmbeddingSample(true);
    try {
      for (const sample of SAMPLE_GUIDELINES) {
        const texts = sample.chunks.map((c) => c.text);
        let embeddings: number[][] = [];
        try {
          const embedRes = await embedTexts(texts, sample.doc.name);
          embeddings = embedRes.embeddings;
        } catch (e) {
          console.warn('Embedding API error or missing API key, storing chunks without embeddings for now:', e);
        }

        const chunksWithEmbeddings: MedicalChunk[] = sample.chunks.map((c, idx) => ({
          ...c,
          embedding: embeddings[idx] || [],
        }));

        await saveDocument(sample.doc, chunksWithEmbeddings);
      }
      const updatedDocs = await getDocuments();
      setDocuments(updatedDocs);
    } catch (err) {
      console.error('Failed to init sample guidelines:', err);
    } finally {
      setIsEmbeddingSample(false);
    }
  }

  async function handleAddCustomDoc(e: React.FormEvent) {
    e.preventDefault();
    if (!customDocTitle.trim() || !customDocContent.trim()) return;

    setIsAddingDoc(true);
    try {
      // Chunk content by paragraphs or 250 words
      const rawParagraphs = customDocContent
        .split(/\n\s*\n/)
        .map((p) => p.trim())
        .filter((p) => p.length > 20);

      const paragraphs = rawParagraphs.length > 0 ? rawParagraphs : [customDocContent.trim()];
      const docId = `doc-${Date.now()}`;

      let embeddings: number[][] = [];
      try {
        const embedRes = await embedTexts(paragraphs, customDocTitle);
        embeddings = embedRes.embeddings;
      } catch (err) {
        console.warn('Embedding API unavailable, storing without embeddings:', err);
      }

      const chunks: MedicalChunk[] = paragraphs.map((text, idx) => ({
        id: `${docId}-${idx}`,
        docId,
        docName: customDocTitle,
        text,
        embedding: embeddings[idx] || [],
      }));

      const newDoc: MedicalDoc = {
        id: docId,
        name: customDocTitle,
        type: 'User Document',
        uploadedAt: new Date().toISOString(),
        chunksCount: chunks.length,
      };

      await saveDocument(newDoc, chunks);
      setCustomDocTitle('');
      setCustomDocContent('');
      await loadDocs();
    } catch (err: any) {
      alert(`Error saving document: ${err.message}`);
    } finally {
      setIsAddingDoc(false);
    }
  }

  async function handleDeleteDoc(id: string) {
    if (!confirm('Are you sure you want to delete this document from IndexedDB?')) return;
    await deleteDocument(id);
    await loadDocs();
  }

  function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      setImageAttachment({
        base64,
        mimeType: file.type || 'image/jpeg',
        preview: URL.createObjectURL(file),
      });
    };
    reader.readAsDataURL(file);
  }

  async function handleSendMessage(presetPrompt?: string) {
    const textToSend = presetPrompt || queryInput.trim();
    if (!textToSend && !imageAttachment) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: textToSend,
      imageUrl: imageAttachment?.preview,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQueryInput('');
    setIsGenerating(true);

    const attachedImg = imageAttachment
      ? { base64: imageAttachment.base64, mimeType: imageAttachment.mimeType }
      : undefined;
    setImageAttachment(null);

    // 1. RAG Retrieval Phase
    let retrievedChunks: RetrievedChunk[] = [];
    try {
      const queryEmbedRes = await embedTexts([textToSend], undefined, true);
      const queryVector = queryEmbedRes.embeddings[0];
      if (queryVector) {
        const initialRetrieved = await searchChunksByVector(queryVector, 5);
        if (initialRetrieved.length > 0) {
          // 2. Gemini Re-ranking Phase
          const rerankRes = await rerankPassages(
            textToSend,
            initialRetrieved.map((c) => c.text),
            'gemini-3.1-flash-lite'
          );
          if (rerankRes.success && rerankRes.scores) {
            retrievedChunks = initialRetrieved.map((chunk, idx) => ({
              ...chunk,
              rerankScore: rerankRes.scores[idx] ?? 5,
            }));
            // Sort by rerank score descending
            retrievedChunks.sort((a, b) => (b.rerankScore || 0) - (a.rerankScore || 0));
          } else {
            retrievedChunks = initialRetrieved;
          }
        }
      }
    } catch (e) {
      console.warn('RAG vector retrieval skipped or failed:', e);
    }

    setCurrentRetrievedPassages(retrievedChunks);

    // 3. Clinical Synthesis Phase with Gemini
    const systemInstruction = `You are Medi-RAG, an educational clinical decision-support AI.
Analyze the user's clinical query alongside the retrieved medical guideline excerpts below.
Provide a clear, structured, evidence-based clinical synthesis:
1. Immediate Clinical Assessment / Triage
2. Evidence-Based Management Steps (explicitly referencing the retrieved guidelines with [Citation: Guideline Name])
3. Key Contraindications & Red Flags
4. Necessary Diagnostic Follow-up

Always maintain an objective, academic tone and emphasize that recommendations must be corroborated by a licensed physician.

Retrieved Guideline Knowledge:
${
  retrievedChunks.length > 0
    ? retrievedChunks.map((c, i) => `[Source ${i + 1}: ${c.docName}]\n${c.text}`).join('\n\n')
    : 'No specific matching local guideline found. Rely on general verified medical evidence.'
}`;

    const assistantMsgId = `asst-${Date.now()}`;
    const assistantMessage: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      modelUsed: selectedModel,
      citations: retrievedChunks,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, assistantMessage]);

    if (useStreaming) {
      let accumulated = '';
      try {
        await streamContent(
          {
            model: selectedModel,
            contents: textToSend,
            systemInstruction,
            temperature: 0.2,
            image: attachedImg,
          },
          {
            onStart: (model) => {
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, modelUsed: model } : m))
              );
            },
            onChunk: (chunk) => {
              accumulated += chunk;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, content: accumulated } : m))
              );
            },
            onFallback: (notice) => {
              accumulated += `\n\n> *Notice: ${notice}*\n\n`;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, content: accumulated } : m))
              );
            },
            onError: (err) => {
              accumulated += `\n\n**Error:** ${err}`;
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, content: accumulated } : m))
              );
              setIsGenerating(false);
            },
            onDone: (finalModel) => {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, modelUsed: finalModel || m.modelUsed }
                    : m
                )
              );
              setIsGenerating(false);
            },
          }
        );
      } catch (err: any) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  content:
                    accumulated ||
                    `Error communicating with Gemini: ${err.message}. Please verify GEMINI_API_KEY.`,
                }
              : m
          )
        );
        setIsGenerating(false);
      }
    } else {
      // Unary Generation
      try {
        const response = await generateContent({
          model: selectedModel,
          contents: textToSend,
          systemInstruction,
          temperature: 0.2,
          image: attachedImg,
        });

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  content: response.text || 'No response generated.',
                  modelUsed: response.modelUsed,
                  latency: response.latency,
                  fallbackOccurred: response.fallbackOccurred,
                }
              : m
          )
        );
      } catch (err: any) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: `Error: ${err.message}` }
              : m
          )
        );
      } finally {
        setIsGenerating(false);
      }
    }
  }

  async function handleRunTriage() {
    setIsTriaging(true);
    setTriageResult('');

    const prompt = `Perform an emergency medical triage and differential diagnosis for this presentation:
Patient: ${triageAge}yo ${triageSex}
Chief Complaint: ${triageComplaint}
Vital Signs: ${triageVitals}
Observed Red Flags: ${triageRedFlags}

Provide:
1. Triage Acuity Level (Resuscitation / Emergent / Urgent / Semi-Urgent)
2. Top 3 Differential Diagnoses with clinical rationale
3. Critical "Do Not Miss" Life Threats
4. Immediate Diagnostic & Stabilization Checklist`;

    try {
      const res = await generateContent({
        model: selectedModel,
        contents: prompt,
        temperature: 0.1,
      });
      setTriageResult(res.text || 'No output received.');
    } catch (err: any) {
      setTriageResult(`Error generating triage: ${err.message}`);
    } finally {
      setIsTriaging(false);
    }
  }

  async function handleTestError(errorType: '429' | '503') {
    setTestResult('Triggering simulated error...');
    try {
      const res = await simulateError(errorType);
      setTestResult(JSON.stringify(res, null, 2));
    } catch (err: any) {
      setTestResult(`Response: ${err.message}`);
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* Top Disclaimer Banner */}
      <div className="bg-amber-500/10 border-b border-amber-200 px-4 py-2 text-xs text-amber-900 flex items-center justify-between">
        <div className="flex items-center gap-2 max-w-5xl mx-auto w-full">
          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
          <span>
            <strong>Educational & Clinical Demonstration Use Only:</strong> Medi-RAG is a decision-support prototype powered by Google Gemini and local IndexedDB retrieval. Never delay or alter professional medical care based on this tool.
          </span>
        </div>
      </div>

      {/* Main Navigation Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-teal-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-teal-500/20">
              <Stethoscope className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-lg text-slate-900 tracking-tight">Medi-RAG</h1>
                <span className="px-2 py-0.5 text-[11px] font-semibold bg-teal-50 text-teal-700 border border-teal-200 rounded-full">
                  Decision Support
                </span>
              </div>
              <p className="text-xs text-slate-500">IndexedDB RAG + Gemini Clinical Reasoning</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                activeTab === 'chat'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Brain className="w-4 h-4 text-indigo-600" />
              <span>Clinical RAG</span>
            </button>
            <button
              onClick={() => setActiveTab('knowledge')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                activeTab === 'knowledge'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BookOpen className="w-4 h-4 text-emerald-600" />
              <span>Guidelines ({documents.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('triage')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                activeTab === 'triage'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Activity className="w-4 h-4 text-rose-600" />
              <span>Triage Lab</span>
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                activeTab === 'settings'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Settings className="w-4 h-4 text-slate-600" />
              <span>System & Lab</span>
            </button>
          </nav>

          {/* Health Badge & Model Selector */}
          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-1.5 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  health?.hasApiKey ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-600">
                {health?.hasApiKey ? 'API Ready' : 'Key Missing'}
              </span>
            </div>

            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="text-xs bg-slate-100 border border-slate-300 rounded-lg px-2.5 py-1.5 font-medium text-slate-700 outline-none focus:ring-2 focus:ring-teal-500"
            >
              {models.length > 0 ? (
                models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.displayName || m.name}
                  </option>
                ))
              ) : (
                <>
                  <option value="gemini-3.8-flash">Gemini 3.8 Flash</option>
                  <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                  <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash Lite</option>
                </>
              )}
            </select>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6 flex flex-col">
        {/* TAB 1: CLINICAL RAG CHAT */}
        {activeTab === 'chat' && (
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-[600px]">
            {/* Left: Chat Flow */}
            <div className="lg:col-span-3 flex flex-col bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              {/* Messages list */}
              <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${
                      msg.role === 'user' ? 'items-end' : 'items-start'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1 text-[11px] text-slate-400">
                      <span>{msg.role === 'user' ? 'Clinical Clinician' : 'Medi-RAG Assistant'}</span>
                      {msg.modelUsed && (
                        <span className="px-1.5 py-0.2 rounded bg-indigo-50 text-indigo-700 font-mono text-[10px]">
                          {msg.modelUsed}
                        </span>
                      )}
                      <span>{msg.timestamp}</span>
                    </div>

                    <div
                      className={`max-w-[85%] rounded-2xl p-4 text-sm leading-relaxed ${
                        msg.role === 'user'
                          ? 'bg-gradient-to-r from-teal-600 to-indigo-600 text-white rounded-tr-none'
                          : 'bg-slate-100 text-slate-800 rounded-tl-none border border-slate-200'
                      }`}
                    >
                      {msg.imageUrl && (
                        <div className="mb-3">
                          <img
                            src={msg.imageUrl}
                            alt="Attached clinical finding"
                            className="max-h-48 rounded-lg object-contain bg-black/10 border border-white/20"
                          />
                        </div>
                      )}
                      <div className="whitespace-pre-wrap">{msg.content}</div>

                      {/* Citations chip bar */}
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-slate-200/60">
                          <p className="text-[11px] font-semibold text-slate-500 mb-1.5 flex items-center gap-1">
                            <BookOpen className="w-3.5 h-3.5 text-teal-600" />
                            Retrieved Guideline Citations ({msg.citations.length}):
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {msg.citations.map((c, idx) => (
                              <span
                                key={idx}
                                className="inline-flex items-center gap-1 text-[11px] bg-white px-2 py-1 rounded-md border border-slate-200 shadow-2xs text-slate-700 font-medium"
                                title={c.text}
                              >
                                <span className="w-4 h-4 rounded-full bg-teal-100 text-teal-800 font-bold flex items-center justify-center text-[9px]">
                                  {idx + 1}
                                </span>
                                {c.docName}
                                {c.rerankScore !== undefined && (
                                  <span className="text-teal-600 font-semibold text-[10px]">
                                    (Relevance {c.rerankScore}/10)
                                  </span>
                                )}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="flex items-center gap-2 text-xs text-slate-500 italic p-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-teal-600" />
                    <span>Retrieving local guidelines, reranking with Gemini, and synthesizing answer...</span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 bg-slate-50 border-t border-slate-200">
                {/* Image preview chip */}
                {imageAttachment && (
                  <div className="mb-2 flex items-center gap-2 bg-indigo-50 border border-indigo-200 p-1.5 rounded-lg w-fit text-xs text-indigo-800">
                    <img
                      src={imageAttachment.preview}
                      alt="Thumbnail"
                      className="w-8 h-8 rounded object-cover"
                    />
                    <span>Multimodal image attached</span>
                    <button
                      onClick={() => setImageAttachment(null)}
                      className="text-indigo-600 hover:text-indigo-900 ml-1 font-bold"
                    >
                      ×
                    </button>
                  </div>
                )}

                <div className="flex items-center gap-2">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    title="Attach clinical image (ECG, X-Ray, lesion photo)"
                    className="p-2.5 rounded-xl border border-slate-300 hover:bg-slate-200/60 text-slate-600 transition"
                  >
                    <ImageIcon className="w-5 h-5" />
                  </button>

                  <input
                    type="text"
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                    placeholder="Ask a clinical question (e.g. 'Acute severe asthma management with GINA', 'Stage 2 HTN dual therapy')..."
                    className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition"
                  />

                  <button
                    type="button"
                    disabled={isGenerating || (!queryInput.trim() && !imageAttachment)}
                    onClick={() => handleSendMessage()}
                    className="bg-gradient-to-r from-teal-600 to-indigo-600 hover:from-teal-700 hover:to-indigo-700 disabled:opacity-50 text-white px-4 py-2.5 rounded-xl font-medium text-sm flex items-center gap-2 shadow-sm transition"
                  >
                    <Send className="w-4 h-4" />
                    <span>Ask RAG</span>
                  </button>
                </div>

                {/* Preset Prompt Buttons */}
                <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="text-slate-400 font-medium mr-1">Quick Scenarios:</span>
                  <button
                    onClick={() =>
                      handleSendMessage(
                        'What is the immediate pharmacological management for acute severe asthma exacerbation according to GINA?'
                      )
                    }
                    className="px-2.5 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-600 transition text-[11px]"
                  >
                    Acute Asthma Exacerbation
                  </button>
                  <button
                    onClick={() =>
                      handleSendMessage(
                        'A 62-year-old with Type 2 Diabetes and established ASCVD: what first-line agent is indicated per ADA guidelines?'
                      )
                    }
                    className="px-2.5 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-600 transition text-[11px]"
                  >
                    T2DM with ASCVD
                  </button>
                  <button
                    onClick={() =>
                      handleSendMessage(
                        'Patient has Stage 2 Hypertension with BP 155/95 mmHg. Which dual-agent classes should be initiated?'
                      )
                    }
                    className="px-2.5 py-1 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg text-slate-600 transition text-[11px]"
                  >
                    Stage 2 Hypertension Dual Therapy
                  </button>
                </div>
              </div>
            </div>

            {/* Right: RAG Live Inspector Panel */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
                <div className="flex items-center gap-1.5">
                  <Brain className="w-4 h-4 text-indigo-600" />
                  <h3 className="font-semibold text-xs text-slate-900 uppercase tracking-wider">
                    RAG Inspection Stream
                  </h3>
                </div>
                <span className="text-[10px] font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded">
                  Top-k Passages
                </span>
              </div>

              <div className="flex-1 overflow-y-auto space-y-3 pr-1 text-xs">
                {currentRetrievedPassages.length === 0 ? (
                  <div className="text-center py-12 text-slate-400">
                    <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No active retrieval query.</p>
                    <p className="text-[11px] mt-1 text-slate-400">
                      Retrieved chunks and Gemini re-ranking scores will appear here.
                    </p>
                  </div>
                ) : (
                  currentRetrievedPassages.map((chunk, idx) => (
                    <div
                      key={chunk.id || idx}
                      className="p-3 rounded-xl bg-slate-50 border border-slate-200 flex flex-col gap-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-800 truncate max-w-[160px]">
                          #{idx + 1} {chunk.docName}
                        </span>
                        {chunk.rerankScore !== undefined && (
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                              chunk.rerankScore >= 7
                                ? 'bg-emerald-100 text-emerald-800'
                                : chunk.rerankScore >= 4
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-slate-200 text-slate-700'
                            }`}
                          >
                            Score: {chunk.rerankScore}/10
                          </span>
                        )}
                      </div>
                      <p className="text-slate-600 leading-normal text-[11px]">{chunk.text}</p>
                      {chunk.similarity !== undefined && (
                        <div className="text-[10px] text-slate-400 font-mono">
                          Vector Sim: {(chunk.similarity * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: KNOWLEDGE BASE (INDEXEDDB) */}
        {activeTab === 'knowledge' && (
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
              <div>
                <h2 className="text-base font-bold text-slate-900">Local IndexedDB Knowledge Base</h2>
                <p className="text-xs text-slate-500 mt-1">
                  Clinical guidelines and reference texts are embedded using Google Gemini and indexed directly in your browser's persistent IndexedDB storage.
                </p>
              </div>
              <button
                type="button"
                disabled={isEmbeddingSample}
                onClick={initSampleKnowledge}
                className="inline-flex items-center gap-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 px-4 py-2 rounded-xl text-xs font-semibold transition"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isEmbeddingSample ? 'animate-spin' : ''}`} />
                <span>Reload Default Guidelines</span>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 Cols: Stored Documents List */}
              <div className="lg:col-span-2 space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Indexed Guidelines & Documents ({documents.length})
                </h3>

                {documents.length === 0 ? (
                  <div className="bg-white p-8 rounded-2xl border border-slate-200 text-center text-slate-400">
                    <BookOpen className="w-10 h-10 mx-auto mb-2 text-slate-300" />
                    <p className="text-sm">No guidelines indexed yet.</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Click "Reload Default Guidelines" or upload a custom medical text on the right.
                    </p>
                  </div>
                ) : (
                  documents.map((doc) => (
                    <div
                      key={doc.id}
                      className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs flex items-center justify-between hover:border-teal-300 transition"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-teal-50 text-teal-600 flex items-center justify-center">
                          <FileText className="w-5 h-5" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-sm text-slate-800">{doc.name}</h4>
                          <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                            <span>{doc.type}</span>
                            <span>•</span>
                            <span>{doc.chunksCount} embedded chunks</span>
                          </div>
                        </div>
                      </div>

                      <button
                        onClick={() => handleDeleteDoc(doc.id)}
                        title="Delete from IndexedDB"
                        className="p-2 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))
                )}
              </div>

              {/* Right Col: Add Custom Medical Guideline */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                <h3 className="text-sm font-bold text-slate-900 mb-1 flex items-center gap-2">
                  <Upload className="w-4 h-4 text-teal-600" />
                  Index Custom Medical Guideline
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                  Add local clinical institutional protocols or case notes. Texts are chunked and vector-embedded automatically.
                </p>

                <form onSubmit={handleAddCustomDoc} className="space-y-3 flex-1 flex flex-col">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Guideline / Document Title
                    </label>
                    <input
                      type="text"
                      required
                      value={customDocTitle}
                      onChange={(e) => setCustomDocTitle(e.target.value)}
                      placeholder="e.g. Sepsis Surviving Campaign 2024"
                      className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 outline-none focus:ring-2 focus:ring-teal-500"
                    />
                  </div>

                  <div className="flex-1 flex flex-col">
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Guideline Content (Paragraphs)
                    </label>
                    <textarea
                      required
                      rows={6}
                      value={customDocContent}
                      onChange={(e) => setCustomDocContent(e.target.value)}
                      placeholder="Paste clinical text here. Paragraphs separated by blank lines will each become an indexed retrieval chunk..."
                      className="w-full flex-1 text-xs bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-teal-500"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={isAddingDoc}
                    className="w-full bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white font-medium text-xs py-2.5 rounded-lg shadow-sm transition flex items-center justify-center gap-2"
                  >
                    {isAddingDoc ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>Embedding with Gemini...</span>
                      </>
                    ) : (
                      <>
                        <Upload className="w-3.5 h-3.5" />
                        <span>Embed & Index Document</span>
                      </>
                    )}
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: TRIAGE & DIFFERENTIAL LAB */}
        {activeTab === 'triage' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Form */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <div className="border-b border-slate-200 pb-3">
                <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-rose-600" />
                  Structured Clinical Triage Assistant
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Enter patient vital signs and acute presentation to generate structured differential diagnosis and acuity scoring.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Age</label>
                  <input
                    type="text"
                    value={triageAge}
                    onChange={(e) => setTriageAge(e.target.value)}
                    className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 outline-none focus:ring-2 focus:ring-teal-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Sex</label>
                  <select
                    value={triageSex}
                    onChange={(e) => setTriageSex(e.target.value)}
                    className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 outline-none focus:ring-2 focus:ring-teal-500"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Chief Complaint</label>
                <textarea
                  rows={2}
                  value={triageComplaint}
                  onChange={(e) => setTriageComplaint(e.target.value)}
                  className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Vital Signs</label>
                <input
                  type="text"
                  value={triageVitals}
                  onChange={(e) => setTriageVitals(e.target.value)}
                  className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Clinical Red Flags & Exam Findings
                </label>
                <textarea
                  rows={2}
                  value={triageRedFlags}
                  onChange={(e) => setTriageRedFlags(e.target.value)}
                  className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg p-2 outline-none focus:ring-2 focus:ring-teal-500"
                />
              </div>

              <button
                type="button"
                disabled={isTriaging}
                onClick={handleRunTriage}
                className="w-full bg-rose-600 hover:bg-rose-700 disabled:opacity-50 text-white font-medium text-xs py-2.5 rounded-xl shadow-sm transition flex items-center justify-center gap-2"
              >
                {isTriaging ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Analyzing differential with Gemini...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Generate Acuity & Differential</span>
                  </>
                )}
              </button>
            </div>

            {/* Output Panel */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
              <h3 className="text-sm font-bold text-slate-900 pb-3 border-b border-slate-200 mb-4 flex items-center gap-2">
                <Brain className="w-4 h-4 text-indigo-600" />
                Differential & Triage Output
              </h3>

              <div className="flex-1 overflow-y-auto">
                {triageResult ? (
                  <div className="text-xs leading-relaxed text-slate-800 whitespace-pre-wrap font-sans bg-slate-50 p-4 rounded-xl border border-slate-200">
                    {triageResult}
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400 py-16">
                    <Activity className="w-10 h-10 mb-2 text-slate-300" />
                    <p className="text-xs">Awaiting patient parameters.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: SYSTEM & MODEL TEST LAB */}
        {activeTab === 'settings' && (
          <div className="space-y-6 max-w-4xl mx-auto w-full">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <h2 className="text-base font-bold text-slate-900 mb-1">Vercel & Gemini System Health</h2>
              <p className="text-xs text-slate-500 mb-4">
                Verify backend API endpoints, environment variable secrets, and error recovery policies.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-slate-700">API Gateway Status</span>
                    <span
                      className={`text-[11px] font-bold px-2 py-0.5 rounded ${
                        health?.status === 'ok'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-rose-100 text-rose-800'
                      }`}
                    >
                      {health?.status === 'ok' ? 'Online' : 'Offline'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500">
                    Express serverless function mounted at <code className="text-indigo-600">/api/*</code>.
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-slate-700">Gemini Secret (GEMINI_API_KEY)</span>
                    <span
                      className={`text-[11px] font-bold px-2 py-0.5 rounded ${
                        health?.hasApiKey
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}
                    >
                      {health?.hasApiKey ? 'Configured' : 'Missing'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500">
                    Add <code className="text-indigo-600">GEMINI_API_KEY</code> to the Vercel Dashboard Environment Variables.
                  </p>
                </div>
              </div>

              {/* Error Simulation Lab */}
              <div className="mt-6 pt-6 border-t border-slate-200">
                <h3 className="text-sm font-bold text-slate-900 mb-1">Simulated Error & Fallback Lab</h3>
                <p className="text-xs text-slate-500 mb-3">
                  Test the server's error mapping for HTTP 429 (Rate Limit) and HTTP 503 (Model Overload).
                </p>

                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleTestError('429')}
                    className="px-3 py-1.5 bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-300 rounded-lg text-xs font-semibold transition"
                  >
                    Simulate 429 Rate Limit
                  </button>
                  <button
                    onClick={() => handleTestError('503')}
                    className="px-3 py-1.5 bg-purple-50 hover:bg-purple-100 text-purple-800 border border-purple-300 rounded-lg text-xs font-semibold transition"
                  >
                    Simulate 503 High Demand
                  </button>
                </div>

                {testResult && (
                  <pre className="mt-3 p-3 bg-slate-900 text-emerald-400 text-xs rounded-xl overflow-x-auto font-mono">
                    {testResult}
                  </pre>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

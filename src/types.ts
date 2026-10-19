export interface MedicalDoc {
  id: string;
  name: string;
  type: string;
  uploadedAt: string;
  chunksCount: number;
}

export interface MedicalChunk {
  id: string;
  docId: string;
  docName: string;
  text: string;
  embedding?: number[];
}

export interface RetrievedChunk extends MedicalChunk {
  similarity: number;
  rerankScore?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  modelUsed?: string;
  timestamp: string;
  citations?: RetrievedChunk[];
  imageUrl?: string;
  latency?: number;
  fallbackOccurred?: boolean;
}

export interface GeminiModelInfo {
  name: string;
  displayName?: string;
}

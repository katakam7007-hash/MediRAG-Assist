import { MedicalDoc, MedicalChunk, RetrievedChunk } from './types';

const DB_NAME = 'medi_rag_db';
const DB_VERSION = 1;
const DOCS_STORE = 'documents';
const CHUNKS_STORE = 'chunks';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(DOCS_STORE)) {
        db.createObjectStore(DOCS_STORE, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(CHUNKS_STORE)) {
        const chunkStore = db.createObjectStore(CHUNKS_STORE, { keyPath: 'id' });
        chunkStore.createIndex('docId', 'docId', { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveDocument(doc: MedicalDoc, chunks: MedicalChunk[]): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([DOCS_STORE, CHUNKS_STORE], 'readwrite');
    const docStore = tx.objectStore(DOCS_STORE);
    const chunkStore = tx.objectStore(CHUNKS_STORE);

    docStore.put(doc);
    for (const chunk of chunks) {
      chunkStore.put(chunk);
    }

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getDocuments(): Promise<MedicalDoc[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DOCS_STORE, 'readonly');
    const store = tx.objectStore(DOCS_STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

export async function deleteDocument(docId: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([DOCS_STORE, CHUNKS_STORE], 'readwrite');
    const docStore = tx.objectStore(DOCS_STORE);
    const chunkStore = tx.objectStore(CHUNKS_STORE);

    docStore.delete(docId);

    const index = chunkStore.index('docId');
    const req = index.openCursor(IDBKeyRange.only(docId));
    req.onsuccess = (e) => {
      const cursor = (e.target as IDBRequest).result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getAllChunks(): Promise<MedicalChunk[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS_STORE, 'readonly');
    const store = tx.objectStore(CHUNKS_STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function searchChunksByVector(queryVector: number[], topK = 4): Promise<RetrievedChunk[]> {
  const chunks = await getAllChunks();
  const scored = chunks
    .filter((c) => c.embedding && c.embedding.length > 0)
    .map((c) => ({
      ...c,
      similarity: cosineSimilarity(queryVector, c.embedding!),
    }))
    .sort((a, b) => b.similarity - a.similarity);

  return scored.slice(0, topK);
}

// Sample initial guidelines
export const SAMPLE_GUIDELINES = [
  {
    doc: {
      id: 'guideline-asthma',
      name: 'GINA 2024 Asthma Management Summary',
      type: 'Clinical Guideline',
      uploadedAt: new Date().toISOString(),
      chunksCount: 3,
    },
    chunks: [
      {
        id: 'asthma-1',
        docId: 'guideline-asthma',
        docName: 'GINA 2024 Asthma Management Summary',
        text: 'Asthma diagnosis relies on a history of variable respiratory symptoms (wheeze, shortness of breath, chest tightness, and cough) and demonstrated expiratory airflow limitation with significant bronchodilator reversibility (FEV1 increase >12% and >200 mL).',
      },
      {
        id: 'asthma-2',
        docId: 'guideline-asthma',
        docName: 'GINA 2024 Asthma Management Summary',
        text: 'GINA Track 1 recommends inhaled corticosteroid (ICS)-formoterol as the preferred reliever across all asthma severity steps. Low-dose ICS-formoterol as-needed reduces severe exacerbation risk by over 60% compared to SABA monotherapy.',
      },
      {
        id: 'asthma-3',
        docId: 'guideline-asthma',
        docName: 'GINA 2024 Asthma Management Summary',
        text: 'For acute severe asthma exacerbations, administer high-concentration oxygen via mask (target SpO2 93-95% in adults), repetitive nebulized SABA with ipratropium bromide, and prompt systemic corticosteroids (oral prednisolone 40-50 mg daily for 5-7 days).',
      },
    ],
  },
  {
    doc: {
      id: 'guideline-diabetes',
      name: 'ADA 2024 Standards of Care: Type 2 Diabetes',
      type: 'Clinical Guideline',
      uploadedAt: new Date().toISOString(),
      chunksCount: 3,
    },
    chunks: [
      {
        id: 'dm-1',
        docId: 'guideline-diabetes',
        docName: 'ADA 2024 Standards of Care: Type 2 Diabetes',
        text: 'Diagnosis of Type 2 Diabetes requires HbA1c >= 6.5%, Fasting Plasma Glucose >= 126 mg/dL (7.0 mmol/L), or 2-hour 75g OGTT >= 200 mg/dL, confirmed by repeated testing in the absence of unequivocal hyperglycemia.',
      },
      {
        id: 'dm-2',
        docId: 'guideline-diabetes',
        docName: 'ADA 2024 Standards of Care: Type 2 Diabetes',
        text: 'First-line pharmacotherapy typically includes Metformin combined with comprehensive lifestyle modification. In patients with established atherosclerotic cardiovascular disease (ASCVD), heart failure, or CKD, an SGLT2 inhibitor and/or GLP-1 receptor agonist with proven CV/renal benefit is indicated regardless of baseline HbA1c.',
      },
      {
        id: 'dm-3',
        docId: 'guideline-diabetes',
        docName: 'ADA 2024 Standards of Care: Type 2 Diabetes',
        text: 'Diabetic Ketoacidosis (DKA) management requires immediate IV fluid resuscitation (normal saline 1-1.5 L in the first hour), continuous IV regular insulin (0.1 units/kg/h), and proactive potassium repletion when serum K+ falls below 5.2 mEq/L before or during insulin administration.',
      },
    ],
  },
  {
    doc: {
      id: 'guideline-hypertension',
      name: 'AHA/ACC Hypertension Clinical Practice Guideline',
      type: 'Clinical Guideline',
      uploadedAt: new Date().toISOString(),
      chunksCount: 2,
    },
    chunks: [
      {
        id: 'htn-1',
        docId: 'guideline-hypertension',
        docName: 'AHA/ACC Hypertension Clinical Practice Guideline',
        text: 'Blood pressure categories: Normal is <120/<80 mmHg. Elevated is 120-129/<80 mmHg. Stage 1 Hypertension is 130-139 or 80-89 mmHg. Stage 2 Hypertension is systolic >=140 or diastolic >=90 mmHg. Treatment target for most adults is <130/80 mmHg.',
      },
      {
        id: 'htn-2',
        docId: 'guideline-hypertension',
        docName: 'AHA/ACC Hypertension Clinical Practice Guideline',
        text: 'For Stage 2 hypertension with BP >20/10 mmHg above goal, prompt initiation of 2 first-line agents of different classes is recommended: thiazide diuretic, CCB, and ACE inhibitor or ARB. ACE inhibitors and ARBs should not be combined.',
      },
    ],
  },
];

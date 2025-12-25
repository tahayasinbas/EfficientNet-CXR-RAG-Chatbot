// AI-XRAY ASSIST - Global Constants

// Zoom Configuration
export const ZOOM = {
  MIN: 0.5,
  MAX: 3,
  STEP: 0.25,
  DEFAULT: 1,
};

// Patient Data Constraints
export const PATIENT = {
  AGE_MIN: 0,
  AGE_MAX: 120,
  GENDERS: {
    MALE: 'M',
    FEMALE: 'F',
  },
  POSITIONS: {
    PA: 'PA',
    AP: 'AP',
  },
};

// File Upload Constraints
export const FILE_UPLOAD = {
  ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/jpg'],
  MAX_SIZE_MB: 10,
  MAX_SIZE_BYTES: 10 * 1024 * 1024,
};

// Risk Thresholds
export const RISK_LEVELS = {
  LOW: 0.3,      // < 30%
  MEDIUM: 0.7,   // 30-70%
  HIGH: 0.7,     // > 70%
};

// Loading States
export const LOADING_STEPS = {
  UPLOADING: 'Görüntü yükleniyor...',
  ANALYZING: 'Model analiz ediyor...',
  GENERATING_HEATMAP: 'Heatmap oluşturuluyor...',
  QUERYING_RAG: 'RAG veritabanı sorgulanıyor...',
  COMPLETE: 'Analiz tamamlandı',
};

// API Endpoints (relative paths) - Django requires trailing slashes
export const API_ENDPOINTS = {
  XRAYS: '/xrays/',
  ANALYZE: (id) => `/xrays/${id}/analyze/`,
};

// Disease Names (NIH Chest X-Ray Dataset)
export const DISEASES = [
  'Atelectasis',
  'Cardiomegaly',
  'Effusion',
  'Infiltration',
  'Mass',
  'Nodule',
  'Pneumonia',
  'Pneumothorax',
  'Consolidation',
  'Edema',
  'Emphysema',
  'Fibrosis',
  'Pleural_Thickening',
  'Hernia',
];

// UI Constants
export const UI = {
  ANIMATION_DURATION: 300,
  TOAST_DURATION: 3000,
  DEBOUNCE_DELAY: 300,
};

export default {
  ZOOM,
  PATIENT,
  FILE_UPLOAD,
  RISK_LEVELS,
  LOADING_STEPS,
  API_ENDPOINTS,
  DISEASES,
  UI,
};

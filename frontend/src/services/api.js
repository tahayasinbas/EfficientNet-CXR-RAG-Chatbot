import axios from 'axios';
import { API_ENDPOINTS } from '../constants';

// Environment variable kullanımı
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Axios instance oluştur
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 saniye timeout
});

// Request interceptor - Her istekte çalışır
api.interceptors.request.use(
  (config) => {
    // Token varsa ekle (gelecekte JWT için)
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Her yanıtta çalışır
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Error handling
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;

      switch (status) {
        case 400:
          error.message = data.message || 'Geçersiz istek';
          break;
        case 401:
          error.message = 'Oturum süresi doldu. Lütfen tekrar giriş yapın.';
          // localStorage.removeItem('token');
          // window.location.href = '/login';
          break;
        case 403:
          error.message = 'Bu işlem için yetkiniz yok';
          break;
        case 404:
          error.message = 'İstenen kaynak bulunamadı';
          break;
        case 500:
          error.message = 'Sunucu hatası. Lütfen daha sonra tekrar deneyin.';
          break;
        default:
          error.message = data.message || 'Bir hata oluştu';
      }
    } else if (error.request) {
      // Request yapıldı ama yanıt alınamadı
      error.message = 'Sunucuya ulaşılamıyor. İnternet bağlantınızı kontrol edin.';
    } else {
      // Request oluşturulurken hata
      error.message = error.message || 'Beklenmeyen bir hata oluştu';
    }

    return Promise.reject(error);
  }
);

// API Methods

/**
 * X-Ray görüntüsü yükle
 * @param {FormData} formData - image, age, gender, position içerir
 * @returns {Promise}
 */
export const uploadXRay = async (formData) => {
  try {
    const response = await api.post(API_ENDPOINTS.XRAYS, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
};

/**
 * X-Ray görüntüsünü analiz et
 * @param {number|string} xrayId - X-Ray ID
 * @returns {Promise}
 */
export const analyzeXRay = async (xrayId) => {
  try {
    const response = await api.post(API_ENDPOINTS.ANALYZE(xrayId));
    return response;
  } catch (error) {
    console.error('Analyze error:', error);
    throw error;
  }
};

/**
 * Tüm X-Ray görüntülerini listele
 * @returns {Promise}
 */
export const getXRays = async () => {
  try {
    const response = await api.get(API_ENDPOINTS.XRAYS);
    return response;
  } catch (error) {
    console.error('Get X-Rays error:', error);
    throw error;
  }
};

/**
 * Tek bir X-Ray görüntüsünü getir
 * @param {number|string} xrayId - X-Ray ID
 * @returns {Promise}
 */
export const getXRay = async (xrayId) => {
  try {
    const response = await api.get(`${API_ENDPOINTS.XRAYS}/${xrayId}`);
    return response;
  } catch (error) {
    console.error('Get X-Ray error:', error);
    throw error;
  }
};

/**
 * Chatbot'a mesaj gönder
 * @param {Object} params - { session_id, xray_id, message }
 * @returns {Promise}
 */
export const sendChatMessage = async ({ sessionId, xrayId, message }) => {
  try {
    // RAG sistemi ilk yüklemede uzun sürebilir, timeout'u artıralım
    const response = await api.post('/chat/send/', {
      session_id: sessionId,
      xray_id: xrayId,
      message: message
    }, {
      timeout: 120000 // 2 dakika (ilk yükleme için)
    });
    return response;
  } catch (error) {
    console.error('Chat message error:', error);
    throw error;
  }
};

/**
 * Chat session oluştur
 * @param {number} xrayId - X-Ray ID (optional)
 * @returns {Promise}
 */
export const createChatSession = async (xrayId = null) => {
  try {
    const response = await api.post('/chat/sessions/', {
      xray: xrayId
    });
    return response;
  } catch (error) {
    console.error('Create chat session error:', error);
    throw error;
  }
};

/**
 * Chat session listesini getir
 * @returns {Promise}
 */
export const getChatSessions = async () => {
  try {
    const response = await api.get('/chat/sessions/');
    return response;
  } catch (error) {
    console.error('Get chat sessions error:', error);
    throw error;
  }
};

export default api;

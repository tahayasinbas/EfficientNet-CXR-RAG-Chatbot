import { RISK_LEVELS } from '../constants';
import theme from '../styles/theme';

/**
 * Risk seviyesine göre renk döndür
 * @param {number} confidence - 0-1 arası confidence değeri
 * @returns {string} - Hex renk kodu
 */
export const getRiskColorByConfidence = (confidence) => {
  if (confidence >= RISK_LEVELS.HIGH) return theme.colors.status.critical;
  if (confidence >= RISK_LEVELS.LOW) return theme.colors.status.warning;
  return theme.colors.status.safe;
};

/**
 * Progress bar genişliği hesapla
 * @param {number} confidence - 0-1 arası confidence değeri
 * @returns {string} - Örn: "85%"
 */
export const getProgressWidth = (confidence) => {
  return `${(confidence * 100).toFixed(0)}%`;
};

/**
 * Tarih formatlama
 * @param {string|Date} date - Tarih
 * @returns {string} - Formatlanmış tarih
 */
export const formatDate = (date) => {
  return new Date(date).toLocaleDateString('tr-TR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Debounce fonksiyonu
 * @param {Function} func - Çalıştırılacak fonksiyon
 * @param {number} wait - Bekleme süresi (ms)
 * @returns {Function}
 */
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Deep clone object
 * @param {Object} obj - Kopyalanacak obje
 * @returns {Object}
 */
export const deepClone = (obj) => {
  return JSON.parse(JSON.stringify(obj));
};

export default {
  getRiskColorByConfidence,
  getProgressWidth,
  formatDate,
  debounce,
  deepClone,
};

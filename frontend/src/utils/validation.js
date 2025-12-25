import { FILE_UPLOAD } from '../constants';

/**
 * Dosya validasyonu yap
 * @param {File} file - Yüklenecek dosya
 * @returns {Object} - { valid: boolean, error: string|null }
 */
export const validateFile = (file) => {
  // Dosya yoksa
  if (!file) {
    return { valid: false, error: 'Dosya seçilmedi' };
  }

  // Dosya tipi kontrolü
  if (!FILE_UPLOAD.ALLOWED_TYPES.includes(file.type)) {
    return {
      valid: false,
      error: `Geçersiz dosya formatı. İzin verilen formatlar: ${FILE_UPLOAD.ALLOWED_TYPES.join(', ')}`,
    };
  }

  // Dosya boyutu kontrolü
  if (file.size > FILE_UPLOAD.MAX_SIZE_BYTES) {
    return {
      valid: false,
      error: `Dosya boyutu ${FILE_UPLOAD.MAX_SIZE_MB}MB'dan büyük olamaz`,
    };
  }

  // Dosya adı kontrolü (özel karakterler)
  const invalidChars = /[<>:"/\\|?*]/;
  if (invalidChars.test(file.name)) {
    return {
      valid: false,
      error: 'Dosya adında geçersiz karakterler var',
    };
  }

  return { valid: true, error: null };
};

/**
 * Görüntü boyutlarını kontrol et
 * @param {File} file - Görüntü dosyası
 * @returns {Promise<Object>} - { width: number, height: number }
 */
export const getImageDimensions = (file) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve({
        width: img.width,
        height: img.height,
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Görüntü yüklenemedi'));
    };

    img.src = url;
  });
};

/**
 * Dosya boyutunu okunabilir formata çevir
 * @param {number} bytes - Byte cinsinden boyut
 * @returns {string} - Örn: "2.5 MB"
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

export default {
  validateFile,
  getImageDimensions,
  formatFileSize,
};

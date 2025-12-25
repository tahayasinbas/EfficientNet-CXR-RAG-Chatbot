import React, { useState, useCallback } from 'react';
import PropTypes from 'prop-types';
import { useDropzone } from 'react-dropzone';
import { FiUploadCloud, FiX, FiAlertCircle, FiFile, FiCheckCircle } from 'react-icons/fi';
import { validateFile, formatFileSize } from '../../utils/validation';
import { FILE_UPLOAD } from '../../constants';
import { cn } from '../../lib/utils';

const UploadZone = ({ onImageSelect, selectedImage, onError }) => {
  const [error, setError] = useState(null);

  const handleFileChange = useCallback((file) => {
    setError(null);
    const validation = validateFile(file);

    if (!validation.valid) {
      setError(validation.error);
      if (onError) onError(validation.error);
      onImageSelect(null);
      return;
    }

    const reader = new FileReader();
    reader.onloadend = () => {
      onImageSelect({
        file: file,
        preview: reader.result,
        name: file.name,
        size: file.size,
        type: file.type,
      });
    };
    reader.onerror = () => {
      const errorMsg = 'Dosya okunamadı. Lütfen tekrar deneyin.';
      setError(errorMsg);
      if (onError) onError(errorMsg);
    };
    reader.readAsDataURL(file);
  }, [onImageSelect, onError]);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles[0]) {
      handleFileChange(acceptedFiles[0]);
    }
  }, [handleFileChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: FILE_UPLOAD.ALLOWED_TYPES.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
    multiple: false,
  });

  const handleRemove = (e) => {
    e.stopPropagation();
    setError(null);
    onImageSelect(null);
  };

  if (selectedImage) {
    return (
      <div className="bg-surface-elevated rounded-lg p-2.5 border border-medical-blue/30 relative animate-fade-in">
        <div className="flex items-start gap-2">
          <div className="flex-shrink-0 w-8 h-8 bg-medical-blue/10 rounded-md flex items-center justify-center border border-medical-blue/20">
            <FiFile className="text-medical-blue" size={14} />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-text-primary truncate">{selectedImage.name}</p>
            <div className="flex items-center gap-1 text-xs text-text-tertiary mt-0.5">
              <FiCheckCircle className="text-status-success" size={12} />
              <span className="text-xs">{formatFileSize(selectedImage.size)}</span>
            </div>
          </div>
          <button
            onClick={handleRemove}
            className="flex-shrink-0 text-text-tertiary hover:text-status-error transition-colors p-1"
            aria-label="Dosyayı kaldır"
          >
            <FiX size={14} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <div
        {...getRootProps()}
        className={cn(
          "border-2 border-dashed rounded-lg p-4 text-center transition-all cursor-pointer",
          "border-border-default hover:border-medical-blue",
          isDragActive && "border-medical-blue bg-medical-blue/10",
          error && "border-status-error"
        )}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center text-text-tertiary">
          <FiUploadCloud className={cn("text-2xl mb-1.5", isDragActive && "text-medical-blue")} />
          <p className="text-xs font-semibold text-text-secondary mb-0.5">
            Sürükle veya <span className="text-medical-blue">Seç</span>
          </p>
          <p className="text-xs text-text-tertiary">
            PNG, JPG, DICOM (Max {FILE_UPLOAD.MAX_SIZE_MB}MB)
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-2 p-2 bg-status-error/10 border border-status-error/20 rounded-lg flex items-center gap-1.5 animate-fade-in">
          <FiAlertCircle className="text-status-error flex-shrink-0" size={14} />
          <p className="text-xs text-status-error/90">{error}</p>
        </div>
      )}
    </>
  );
};

UploadZone.propTypes = {
  onImageSelect: PropTypes.func.isRequired,
  selectedImage: PropTypes.object,
  onError: PropTypes.func,
};

export default UploadZone;


import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { cn } from '../../lib/utils';
import { FiChevronDown, FiChevronUp, FiInfo } from 'react-icons/fi';
import DiagnosisDetailModal from './DiagnosisDetailModal';

const getRiskColor = (confidence) => {
  if (confidence >= 0.7) return 'bg-status-error';
  if (confidence >= 0.3) return 'bg-status-warning';
  return 'bg-status-success';
};

const getRiskTextColor = (confidence) => {
  if (confidence >= 0.7) return 'text-status-error';
  if (confidence >= 0.3) return 'text-status-warning';
  return 'text-status-success';
};

const getRiskLevel = (confidence) => {
  if (confidence >= 0.7) return 'Yüksek Risk';
  if (confidence >= 0.3) return 'Orta Risk';
  return 'Düşük Risk';
};

const ProgressBars = ({ results, patientInfo, compact = false }) => {
  const [showAll, setShowAll] = useState(false);
  const [selectedDiagnosis, setSelectedDiagnosis] = useState(null);
  const [showFullModal, setShowFullModal] = useState(false);

  if (!results || results.length === 0) {
    return null;
  }

  // Compact mode: sadece top 3, Normal mode: top 5
  const topCount = compact ? 3 : 5;
  const topResults = results.slice(0, topCount);
  const remainingResults = results.slice(topCount);

  const renderProgressBar = (result, index) => {
    const percentage = (result.confidence * 100).toFixed(1);

    if (compact) {
      return (
        <div key={result.disease_name} className="flex items-center gap-2 text-xs">
          <span className="text-text-tertiary font-medium w-4">{index + 1}.</span>
          <span className="text-text-primary flex-1 truncate">{result.disease_name}</span>
          <span className={cn("font-bold", getRiskTextColor(result.confidence))}>
            {percentage}%
          </span>
        </div>
      );
    }

    return (
      <div
        key={result.disease_name}
        className="group cursor-pointer"
        onClick={() => setSelectedDiagnosis(result)}
      >
        <div className="flex justify-between items-baseline mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-text-primary group-hover:text-medical-blue transition-colors">
              {result.disease_name}
            </span>
            <FiInfo
              size={14}
              className="text-text-tertiary group-hover:text-medical-blue transition-colors opacity-0 group-hover:opacity-100"
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-tertiary">
              {getRiskLevel(result.confidence)}
            </span>
            <span className={cn("text-base font-bold", getRiskTextColor(result.confidence))}>
              {percentage}%
            </span>
          </div>
        </div>
        <div className="w-full bg-bg-tertiary rounded-full h-2.5 overflow-hidden group-hover:h-3 transition-all">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              getRiskColor(result.confidence)
            )}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>
    );
  };

  if (compact) {
    return (
      <div className="space-y-3 animate-fade-in">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
            Analiz Sonuçları
          </h3>
          <span className="text-xs text-text-tertiary">{results.length} hastalık</span>
        </div>

        <div className="space-y-2 bg-surface-elevated rounded-lg p-3">
          {topResults.map((result, index) => renderProgressBar(result, index))}
        </div>

        <button
          onClick={() => setShowFullModal(true)}
          className="w-full px-3 py-2 text-xs font-medium text-medical-blue bg-medical-blue/10 hover:bg-medical-blue/20 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <FiInfo size={14} />
          Tüm Sonuçları Görüntüle
        </button>

        {/* Full Results Modal */}
        {showFullModal && (
          <DiagnosisDetailModal
            diagnosis={topResults[0]}
            allResults={results}
            patientInfo={patientInfo}
            onClose={() => setShowFullModal(false)}
            showAllResults={true}
          />
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Top 5 Sonuçlar */}
      <div className="space-y-4">
        <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
          En Yüksek Olasılıklar
        </h3>
        <div className="space-y-3.5">
          {topResults.map((result, index) => renderProgressBar(result, index))}
        </div>
      </div>

      {/* Diğer Sonuçlar */}
      {remainingResults.length > 0 && (
        <div className="border-t border-border-light pt-4">
          <button
            onClick={() => setShowAll(!showAll)}
            className="flex items-center justify-between w-full text-left mb-3 hover:bg-surface-elevated p-2 -m-2 rounded-lg transition-colors"
          >
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Diğer Hastalıklar ({remainingResults.length})
            </span>
            {showAll ? (
              <FiChevronUp className="text-text-tertiary" size={16} />
            ) : (
              <FiChevronDown className="text-text-tertiary" size={16} />
            )}
          </button>

          {showAll && (
            <div className="space-y-3 animate-fade-in">
              {remainingResults.map((result, index) => renderProgressBar(result, index + topCount))}
            </div>
          )}
        </div>
      )}

      {/* Toplam Hastalık Sayısı */}
      <div className="text-center text-xs text-text-tertiary pt-2 border-t border-border-light">
        Toplam {results.length} hastalık tarandı
      </div>

      {/* Detay Modal */}
      {selectedDiagnosis && (
        <DiagnosisDetailModal
          diagnosis={selectedDiagnosis}
          allResults={results}
          patientInfo={patientInfo}
          onClose={() => setSelectedDiagnosis(null)}
        />
      )}
    </div>
  );
};

ProgressBars.propTypes = {
  results: PropTypes.arrayOf(
    PropTypes.shape({
      disease_name: PropTypes.string.isRequired,
      confidence: PropTypes.number.isRequired,
    })
  ).isRequired,
  patientInfo: PropTypes.shape({
    age: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    gender: PropTypes.string,
    position: PropTypes.string,
  }),
  compact: PropTypes.bool,
};

export default ProgressBars;


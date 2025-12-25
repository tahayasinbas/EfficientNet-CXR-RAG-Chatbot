import React from 'react';
import PropTypes from 'prop-types';
import { FiAlertTriangle } from 'react-icons/fi';
import { cn } from '../../lib/utils';

const getRiskColor = (confidence) => {
  if (confidence >= 0.7) return {
    bg: 'bg-status-error/10',
    text: 'text-status-error',
    progress: 'bg-status-error'
  };
  if (confidence >= 0.3) return {
    bg: 'bg-status-warning/10',
    text: 'text-status-warning',
    progress: 'bg-status-warning'
  };
  return {
    bg: 'bg-status-success/10',
    text: 'text-status-success',
    progress: 'bg-status-success'
  };
};

const StickyResultHeader = ({ topResult }) => {
  if (!topResult) return null;

  const percentage = (topResult.confidence * 100).toFixed(0);
  const risk = getRiskColor(topResult.confidence);

  return (
    <div className="p-4 border-b border-border-light bg-surface flex-shrink-0">
      <div className="flex items-center gap-3 mb-2">
        <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center", risk.bg)}>
          <FiAlertTriangle className={risk.text} size={20} />
        </div>
        <div className="flex-1">
          <h3 className="text-base font-semibold text-text-primary">
            {topResult.disease_name}
          </h3>
          <p className="text-xs text-text-secondary">En Yüksek Olasılıklı Tanı</p>
        </div>
        <p className={cn("text-2xl font-bold", risk.text)}>{percentage}%</p>
      </div>

      <div className="w-full bg-bg-tertiary rounded-full h-2">
        <div
          className={cn("h-2 rounded-full", risk.progress)}
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
};

StickyResultHeader.propTypes = {
  topResult: PropTypes.shape({
    disease_name: PropTypes.string.isRequired,
    confidence: PropTypes.number.isRequired,
  }),
};

export default StickyResultHeader;


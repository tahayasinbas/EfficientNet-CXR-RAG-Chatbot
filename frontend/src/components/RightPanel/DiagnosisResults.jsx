import React from 'react';
import PropTypes from 'prop-types';
import ProgressBars from './ProgressBars';
import { FiCpu, FiCheckCircle } from 'react-icons/fi';
import { cn } from '../../lib/utils';

const StatusCard = ({ icon, title, status, statusColor }) => (
  <div className="flex items-center gap-3 p-3 bg-surface-elevated rounded-lg border border-border-light">
    <div className={cn("flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center", statusColor?.bg)}>
      {icon}
    </div>
    <div>
      <p className="text-sm font-semibold text-text-primary">{title}</p>
      <div className={cn("flex items-center gap-1.5 text-xs", statusColor?.text)}>
        {status.icon}
        <span>{status.label}</span>
      </div>
    </div>
  </div>
);

const DiagnosisResults = ({ results, loading, patientInfo }) => {
  if (loading) {
    return (
      <div className="space-y-3 animate-pulse">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="h-3 bg-bg-tertiary rounded w-1/3"></div>
            <div className="h-3 bg-bg-tertiary rounded w-1/5 ml-auto"></div>
          </div>
        ))}
        <div className="text-center pt-2">
          <p className="text-xs text-text-secondary">Analiz ediliyor...</p>
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="text-center py-4">
        <div className="space-y-2">
          <StatusCard
            icon={<FiCpu className="text-medical-blue" />}
            title="Model: EfficientNet-B4"
            status={{ icon: <FiCheckCircle size={14} />, label: 'Hazır' }}
            statusColor={{ text: 'text-status-success' }}
          />
        </div>
        <p className="text-xs text-text-tertiary mt-4">
          Görüntü yükleyin ve analiz başlatın.
        </p>
      </div>
    );
  }

  return <ProgressBars results={results} patientInfo={patientInfo} compact={true} />;
};

DiagnosisResults.propTypes = {
  results: PropTypes.array,
  loading: PropTypes.bool,
  patientInfo: PropTypes.shape({
    age: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    gender: PropTypes.string,
    position: PropTypes.string,
  }),
};

export default DiagnosisResults;

